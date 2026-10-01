"""Deterministic raw-tensor wire format and bidirectional byte accounting.

The length function uses the same canonical header and dtype/shape checks as
encode_packet. No pickle, compression, storage aliasing, or imaginary cache hit.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import struct
from typing import Mapping

import numpy as np

MAGIC = b"PCL7"
DTYPES = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8"),
          "int32": np.dtype("<i4"), "int64": np.dtype("<i8"),
          "uint8": np.dtype("u1")}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str = "float32"

    @property
    def nbytes(self):
        if self.dtype not in DTYPES or any(x < 0 for x in self.shape):
            raise ValueError("unsupported tensor spec")
        return int(np.prod(self.shape, dtype=np.int64)) * DTYPES[self.dtype].itemsize


def _spec_of(value):
    if isinstance(value, TensorSpec):
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    name = array.dtype.name
    if name not in DTYPES:
        raise ValueError(f"wire tensor dtype {name} unsupported")
    return TensorSpec(tuple(int(x) for x in array.shape), name)


def pack_mask(indices, size):
    if size < 0 or len(set(indices)) != len(indices) or any(i < 0 or i >= size for i in indices):
        raise ValueError("invalid mask")
    raw = bytearray((size + 7)//8)
    for i in indices:
        raw[i//8] |= 1 << (i % 8)
    return bytes(raw)


def unpack_mask(raw, size):
    if len(raw) != (size + 7)//8 or (size % 8 and raw[-1] >> (size % 8)):
        raise ValueError("invalid mask encoding")
    return [i for i in range(size) if raw[i//8] & (1 << (i % 8))]


def _header(kind, version, client_id, tensors, metadata=None, mask=None, mask_size=None):
    if not isinstance(version, int) or version < 0 or not isinstance(client_id, int) or client_id < 0:
        raise ValueError("nonnegative version/client ID required")
    if not isinstance(kind, str) or not kind:
        raise ValueError("packet kind required")
    if mask is None and mask_size is not None or mask is not None and mask_size is None:
        raise ValueError("mask and its size must be supplied together")
    specs = {name: _spec_of(value) for name, value in tensors.items()}
    if any(not isinstance(name, str) or not name for name in specs):
        raise ValueError("named tensors required")
    desc = [{"name": name, "shape": list(specs[name].shape), "dtype": specs[name].dtype,
             "bytes": specs[name].nbytes} for name in sorted(specs)]
    if version > 999999 or client_id > 999999:
        raise ValueError("wire version/client ID exceeds fixed-width field")
    header = {"kind": kind, "version": f"{version:06d}", "client_id": f"{client_id:06d}",
              "tensors": desc, "metadata": metadata or {},
              "mask_size": mask_size,
              "mask": base64.b64encode(pack_mask(mask, mask_size)).decode() if mask is not None else None}
    raw = json.dumps(header, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode()
    return raw, specs


def packet_length(kind, version, client_id, tensors, metadata=None, mask=None, mask_size=None):
    header, specs = _header(kind, version, client_id, tensors, metadata, mask, mask_size)
    return len(MAGIC) + 4 + len(header) + sum(spec.nbytes for spec in specs.values())


def encode_packet(kind, version, client_id, tensors, metadata=None, mask=None, mask_size=None):
    header, specs = _header(kind, version, client_id, tensors, metadata, mask, mask_size)
    payload = bytearray(MAGIC + struct.pack("<I", len(header)) + header)
    for name in sorted(specs):
        value = tensors[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=DTYPES[specs[name].dtype])
        if tuple(array.shape) != specs[name].shape:
            raise ValueError("tensor changed during encoding")
        payload.extend(array.tobytes(order="C"))
    if len(payload) != packet_length(kind, version, client_id, tensors, metadata, mask, mask_size):
        raise AssertionError("wire length mismatch")
    return bytes(payload)


def decode_packet(raw):
    if not raw.startswith(MAGIC) or len(raw) < len(MAGIC)+4:
        raise ValueError("invalid packet magic")
    n = struct.unpack_from("<I", raw, len(MAGIC))[0]
    at = len(MAGIC)+4
    header = json.loads(raw[at:at+n])
    if len(header["version"]) != 6 or len(header["client_id"]) != 6:
        raise ValueError("invalid fixed-width version/client field")
    header["version"] = int(header["version"])
    header["client_id"] = int(header["client_id"])
    at += n
    tensors = {}
    for item in header["tensors"]:
        spec = TensorSpec(tuple(item["shape"]), item["dtype"])
        if spec.nbytes != item["bytes"] or at+spec.nbytes > len(raw):
            raise ValueError("tensor length mismatch")
        tensors[item["name"]] = np.frombuffer(raw[at:at+spec.nbytes],
                                                  dtype=DTYPES[spec.dtype]).reshape(spec.shape).copy()
        at += spec.nbytes
    if at != len(raw):
        raise ValueError("trailing wire bytes")
    if header["mask"] is not None:
        header["mask_indices"] = unpack_mask(base64.b64decode(header["mask"], validate=True),
                                              header["mask_size"])
    return header, tensors


def bert_adapter_specs(rank):
    """BERT-base 12 layers × query/value, 768×768 projections, r columns."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    return {f"bert.encoder.layer.{layer}.attention.self.{target}.lora_{factor}":
            TensorSpec((768, rank) if factor == "b" else (rank, 768))
            for layer in range(12) for target in ("query", "value")
            for factor in ("a", "b")}


def bert_head_specs():
    return {"classifier.weight": TensorSpec((2, 768)),
            "classifier.bias": TensorSpec((2,))}


def full_download_specs(rank):
    return {**bert_adapter_specs(rank), **bert_head_specs()}


def sparse_delta_tensors(before: Mapping, after: Mapping, selected: Mapping[str, list[int]],
                         head_names=("classifier.weight", "classifier.bias")):
    """Extract transmitted updates from real tensors; unselected factors send zero bytes.

    `selected` contains a rank-index list for each actually changed LoRA
    factor. A uses selected rows, B selected columns. Head is always complete.
    The returned per-tensor masks are encoded in metadata by encode_delta.
    """
    if set(before) != set(after):
        raise ValueError("before/after tensor keys differ")
    if not set(selected) <= set(before):
        raise ValueError("selection names absent from model")
    out = {}
    masks = {}
    for name, indices in sorted(selected.items()):
        old, new = np.asarray(before[name]), np.asarray(after[name])
        if old.shape != new.shape or old.ndim != 2:
            raise ValueError("adapter factor shape mismatch")
        if name.endswith(("lora_A", "lora_a")):
            axis, size = 0, old.shape[0]
        elif name.endswith(("lora_B", "lora_b")):
            axis, size = 1, old.shape[1]
        else:
            raise ValueError("selected tensor must be LoRA A or B")
        raw_mask = pack_mask(indices, size)
        out[name] = np.take(new-old, indices, axis=axis)
        masks[name] = {"size": size, "bits": base64.b64encode(raw_mask).decode()}
    for name in head_names:
        if name not in before:
            raise ValueError("head tensor missing")
        out[name] = np.asarray(after[name]) - np.asarray(before[name])
    return out, masks


def encode_sparse_delta(version, client_id, before, after, selected,
                        head_names=("classifier.weight", "classifier.bias")):
    tensors, masks = sparse_delta_tensors(before, after, selected, head_names)
    return encode_packet("delta_upload", version, client_id, tensors,
                         {"rank_masks": masks, "head_names": list(head_names)})


@dataclass
class VersionedClientCache:
    """Only acknowledge a download after send; reuse requires byte-identical content."""
    fingerprints: dict[int, dict[str, str]] = field(default_factory=dict)
    versions: dict[int, int] = field(default_factory=dict)

    def plan(self, client_id, version, tensors, kind="model_download"):
        known = self.fingerprints.get(client_id, {})
        fingerprints = {}
        send = {}
        for name, value in tensors.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            identity = hashlib.sha256(array.tobytes(order="C") +
                                      str(array.shape).encode() + array.dtype.str.encode()).hexdigest()
            fingerprints[name] = identity
            if known.get(name) != identity:
                send[name] = value
        md = {"cache_base_version": self.versions.get(client_id),
              "task_init": client_id not in self.versions,
              "complete_current_state": True}
        packet = encode_packet(kind, version, client_id, send, md)
        return packet, fingerprints

    def acknowledge(self, client_id, version, fingerprints):
        if client_id in self.versions and version < self.versions[client_id]:
            raise ValueError("stale client cache acknowledgement")
        self.versions[client_id] = version
        self.fingerprints[client_id] = dict(fingerprints)

    def require_version(self, client_id, version):
        if self.versions.get(client_id) != version:
            raise ValueError("stale client upload")


def b8_wire_manifest(client_schedule=None, physically_encode=True):
    """Encode an exact 100-round shape list, independent of training results.

    The default deterministic schedule is only for packet lengths: every round
    has ten logical participants, and client IDs 0..49 recur uniformly. Actual
    simulation may use another schedule; its ledger uses its actual IDs.
    """
    if client_schedule is None:
        client_schedule = [tuple((10*(r-1)+j) % 50 for j in range(10))
                           for r in range(1, 101)]
    if len(client_schedule) != 100 or any(len(x) != 10 or len(set(x)) != 10 for x in client_schedule):
        raise ValueError("B8 requires 100 ten-client rounds")
    specs = full_download_specs(8)
    tensors = {name: np.zeros(spec.shape, dtype=DTYPES[spec.dtype]) for name, spec in specs.items()}
    mask_ids = list(range(24 * 2 * 8))
    rounds = []
    seen = set()
    for r, clients in enumerate(client_schedule, 1):
        down = up = 0
        for cid in clients:
            # Task initialization is carried by first normal full download.
            md = {"task_init": cid not in seen, "parent_version": r-1}
            up_md = {"parent_version": r-1, "delta": True}
            down_len = packet_length("model_download", r-1, cid, specs, md)
            up_len = packet_length("delta_upload", r-1, cid, specs, up_md,
                                   mask_ids, len(mask_ids))
            if physically_encode:
                actual_down = encode_packet("model_download", r-1, cid, tensors, md)
                actual_up = encode_packet("delta_upload", r-1, cid, tensors, up_md,
                                          mask_ids, len(mask_ids))
                if len(actual_down) != down_len or len(actual_up) != up_len:
                    raise AssertionError("B8 measured packet differs from predicted wire length")
                down_len, up_len = len(actual_down), len(actual_up)
            down += down_len
            up += up_len
            seen.add(cid)
        rounds.append({"round": r, "download_bytes": down, "upload_bytes": up,
                       "total_bytes": down+up})
    return {"definition": "full_r8_adapter_head_down_both_factor_head_delta_up_100_rounds",
            "rounds": rounds, "total_bytes": sum(x["total_bytes"] for x in rounds),
            "logical_clients": len(seen), "cold_start_w0_excluded": True,
            "physically_encoded_reference": physically_encode,
            "upload_mask_size": len(mask_ids),
            "reference_metadata": {"download": {"task_init": "boolean", "parent_version": "round-1"},
                                   "upload": {"parent_version": "round-1", "delta": True}}}


@dataclass
class WireLedger:
    cap: int
    deployed_prefix: int = 0
    diagnostic_after_budget: int = 0
    budget_stop_round: int | None = None
    events: list[dict] = field(default_factory=list)

    def admit(self, next_round, packet_lengths):
        if any(x < 0 for x in packet_lengths):
            raise ValueError("negative packet length")
        if self.budget_stop_round is not None:
            return False
        if self.deployed_prefix + sum(packet_lengths) > self.cap:
            self.budget_stop_round = next_round - 1
            return False
        return True

    def record(self, event_id, round_number, direction, packet_bytes, diagnostic=False):
        if packet_bytes < 0 or direction not in {"download", "upload", "control"}:
            raise ValueError("invalid wire event")
        if any(x["event_id"] == event_id for x in self.events):
            raise ValueError("duplicate wire event; use a distinct attempt ID for real retransmission")
        bucket = "diagnostic_after_budget" if diagnostic else "deployed_prefix"
        if not diagnostic and getattr(self, bucket) + packet_bytes > self.cap:
            raise ValueError("wire event exceeds budget")
        setattr(self, bucket, getattr(self, bucket) + packet_bytes)
        self.events.append({"event_id": event_id, "round": round_number,
                            "direction": direction, "bytes": packet_bytes, "bucket": bucket})

    def as_dict(self):
        return {"cap": self.cap, "deployed_prefix": self.deployed_prefix,
                "diagnostic_after_budget": self.diagnostic_after_budget,
                "budget_stop_round": self.budget_stop_round, "events": list(self.events)}
