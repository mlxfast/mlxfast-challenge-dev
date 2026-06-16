"""ReferenceSlotBank — stream experts straight from the original checkpoint.

This is the ground-truth counterpart to mlx_models' ExpertSlotBank. Where
ExpertSlotBank reads expert records from the participant-produced
``weights/experts/layer_NN.bin`` files, ReferenceSlotBank reads each expert
directly out of the *unmodified* reference checkpoint's stacked safetensors —
so it needs no offline transform and represents the canonical 4-bit weights.

How it works
────────────
The reference checkpoint stores routed experts as stacked tensors:

    model.layers.{L}.ffn.switch_mlp.{gate,up,down}_proj.{weight,scales}
        shape = (num_experts, *per_expert_shape)   # axis 0 = expert index

Because axis 0 is the expert index and safetensors stores tensors row-major
and contiguously, expert ``e`` occupies a contiguous byte range at

    data_section_start + tensor_begin + e * per_expert_nbytes

so a single ``os.pread`` extracts one expert's slice with zero copying — the
exact same access pattern ExpertSlotBank uses, just pointed at the stacked
checkpoint instead of the repacked per-layer bins.

The returned dict structure matches ExpertSlotBank.get() so the (frozen)
model's StreamingSwitchGLU consumes it unchanged:

    {"gate_proj": {"weight": mx.array, "scales": mx.array},
     "up_proj":   {...},
     "down_proj": {...}}
"""
from __future__ import annotations

import json
import os
import struct
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# safetensors dtype string → numpy dtype string. Experts are U32 (mxfp4
# packed) + U8 (scales); the others are listed for completeness.
_ST_TO_NP = {
    "U8": "|u1", "I8": "|i1",
    "U16": "<u2", "I16": "<i2", "F16": "<f2",
    "U32": "<u4", "I32": "<i4", "F32": "<f4",
    "U64": "<u8", "I64": "<i8", "F64": "<f8",
}

_PROJS = ("gate_proj", "up_proj", "down_proj")
_TTYPES = ("weight", "scales", "biases")


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_section_start_byte_offset)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def _find_checkpoint_dir(base: Path) -> Path:
    if (base / "config.json").exists():
        return base
    for root, _dirs, files in os.walk(base, followlinks=True):
        if "config.json" in files:
            return Path(root)
    raise FileNotFoundError(f"No config.json found under {base}")


class ReferenceSlotBank:
    """Fixed-capacity LRU of expert records read from the stacked checkpoint.

    Drop-in for ExpertSlotBank: same ``get(layer, expert)`` contract, same
    return shape — but the source is the original reference checkpoint, so the
    experts are the canonical (untransformed) 4-bit weights.
    """

    def __init__(self, capacity: int = 128) -> None:
        self.capacity = capacity
        self._lru: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self._fds: dict[str, int] = {}
        # (layer, proj, ttype) -> dict(path, base_offset, nbytes, shape, np_dtype)
        self._slabs: dict[tuple[int, str, str], dict] = {}
        self.num_layers = 0
        self.num_experts = 0

    # ── setup ──────────────────────────────────────────────────────────────
    def configure(self, reference_dir: str) -> "ReferenceSlotBank":
        model_dir = _find_checkpoint_dir(Path(reference_dir))
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            weight_map = {}
            for shard in sorted(model_dir.glob("*.safetensors")):
                hdr, _ = _read_safetensors_header(str(shard))
                for k in hdr:
                    if k != "__metadata__":
                        weight_map[k] = shard.name

        # Cache one parsed header per shard we touch.
        headers: dict[str, tuple[dict, int]] = {}

        def hdr_for(shard_name: str) -> tuple[dict, int]:
            if shard_name not in headers:
                headers[shard_name] = _read_safetensors_header(
                    str(model_dir / shard_name)
                )
            return headers[shard_name]

        layers: set[int] = set()
        n_experts: Optional[int] = None
        for key, shard_name in weight_map.items():
            if ".ffn.switch_mlp." not in key or ".shared_experts." in key:
                continue
            parts = key.split(".")
            layer = int(parts[parts.index("layers") + 1])
            proj = parts[-2]
            ttype = parts[-1]
            if proj not in _PROJS or ttype not in _TTYPES:
                continue
            header, data_start = hdr_for(shard_name)
            info = header[key]
            begin, end = info["data_offsets"]
            shape = info["shape"]
            if not shape:
                continue
            experts = shape[0]
            n_experts = experts if n_experts is None else n_experts
            per_expert_nbytes = (end - begin) // experts
            self._slabs[(layer, proj, ttype)] = {
                "path": str(model_dir / shard_name),
                "base": data_start + begin,
                "nbytes": per_expert_nbytes,
                "shape": list(shape[1:]),
                "dtype": _ST_TO_NP.get(info["dtype"], None),
                "st_dtype": info["dtype"],
            }
            layers.add(layer)

        if not self._slabs:
            raise RuntimeError(
                f"No .ffn.switch_mlp. expert tensors found under {model_dir}"
            )
        self.num_layers = (max(layers) + 1) if layers else 0
        self.num_experts = n_experts or 0
        return self

    # ── access ───────────────────────────────────────────────────────────
    def _fd(self, path: str) -> int:
        if path not in self._fds:
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._fds[path]

    def _load(self, layer_idx: int, expert_idx: int) -> dict:
        result: dict[str, dict[str, mx.array]] = {}
        for proj in _PROJS:
            arrays: dict[str, mx.array] = {}
            for ttype in _TTYPES:
                slab = self._slabs.get((layer_idx, proj, ttype))
                if slab is None:
                    continue
                if slab["dtype"] is None:
                    raise TypeError(
                        f"unsupported expert dtype {slab['st_dtype']} for "
                        f"layer={layer_idx} {proj}.{ttype}"
                    )
                offset = slab["base"] + expert_idx * slab["nbytes"]
                raw = os.pread(self._fd(slab["path"]), slab["nbytes"], offset)
                np_arr = np.frombuffer(raw, dtype=slab["dtype"]).reshape(slab["shape"])
                arrays[ttype] = mx.array(np_arr)
            if arrays:
                result[proj] = arrays
        return result

    def get(self, layer_idx: int, expert_idx: int) -> dict:
        key = (layer_idx, expert_idx)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        record = self._load(layer_idx, expert_idx)
        if len(self._lru) >= self.capacity:
            self._lru.popitem(last=False)
        self._lru[key] = record
        return record

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    @property
    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "resident": len(self._lru),
            "open_files": len(self._fds),
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
        }
