"""The narrow channel between the candidate container and the trusted comparator.

The candidate container computes *observations* -- its own logits, its own loss values,
its own LR trace -- and returns them here. It never returns a verdict, because a verdict
computed inside the candidate's own process is a verdict the candidate can forge. The
trusted comparator owns gold and applies every predicate after the container has exited.

Everything in this module exists because the data crossing this boundary is attacker
controlled. Two rules follow, and neither is negotiable:

  1. **No pickle, ever.** `torch.load` is an unpickler underneath, and PyTorch's own
     documentation says not to point it at untrusted data. `weights_only=True` narrows
     the gadget surface but is a mitigation, not a boundary, and it has been bypassed
     before. Tensors cross as dtype + shape + raw little-endian bytes, decoded by
     `frombuffer`, which cannot execute anything.
  2. **Every field is validated before it is believed.** Size, dtype, rank, element
     count and exact byte length are all checked against declared limits *before* any
     allocation happens, so a candidate cannot exhaust the trusted process's memory by
     declaring a petabyte-shaped tensor.

The wire format is JSON:

    {"observations": {"<name>": <value>}, "errors": {"<name>": "<message>"}}

where `<value>` is a JSON scalar, a (possibly nested) list of scalars, or a tensor:

    {"__tensor__": {"dtype": "float32", "shape": [2, 3], "data": "<base64>"}}
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from typing import Any

# --------------------------------------------------------------------------- #
# Limits. Deliberately generous for honest work and fatal for anything absurd.
# --------------------------------------------------------------------------- #
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
"""Whole-document ceiling, checked before `json.loads` sees a single byte."""

MAX_OBSERVATIONS = 512
MAX_NAME_LEN = 128
MAX_DETAIL_LEN = 2000

MAX_TENSOR_ELEMENTS = 4_000_000
MAX_TENSOR_RANK = 6
MAX_LIST_ELEMENTS = 100_000

# `bool` is 1 byte in torch's numpy-compatible buffer protocol. No float64: nothing in
# the check suite needs it, and a narrower allowlist is a smaller decoder.
ALLOWED_DTYPES: dict[str, int] = {
    "float32": 4,
    "int64": 8,
    "bool": 1,
}


class ProtocolError(ValueError):
    """The candidate's output did not conform. Always fatal, never a check failure.

    Kept distinct from `CheckFailure` on purpose: "the candidate's model is wrong" and
    "the candidate returned something that is not a valid observation document" are
    different events, and collapsing them would let a candidate turn a malformed payload
    into a graded outcome of its choosing.
    """


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _decode_tensor(spec: Any, *, name: str):
    """Decode one `{"__tensor__": ...}` object into a torch tensor.

    Validation order matters: everything cheap and declarative is checked before the
    base64 payload is decoded, and the decoded length is checked before the buffer is
    handed to torch. A candidate declaring `shape: [1e9, 1e9]` is rejected on arithmetic,
    not on an allocation failure.
    """
    import torch

    _require(isinstance(spec, dict), f"{name}: tensor spec must be an object")
    unexpected = set(spec) - {"dtype", "shape", "data"}
    _require(not unexpected, f"{name}: unexpected tensor fields {sorted(unexpected)}")

    dtype = spec.get("dtype")
    _require(dtype in ALLOWED_DTYPES, f"{name}: dtype {dtype!r} is not allowed")

    shape = spec.get("shape")
    _require(isinstance(shape, list), f"{name}: shape must be a list")
    _require(len(shape) <= MAX_TENSOR_RANK, f"{name}: rank {len(shape)} exceeds limit")
    for dim in shape:
        _require(
            isinstance(dim, int) and not isinstance(dim, bool) and 0 <= dim,
            f"{name}: shape entries must be non-negative ints, got {dim!r}",
        )

    n_elements = 1
    for dim in shape:
        n_elements *= dim
        _require(
            n_elements <= MAX_TENSOR_ELEMENTS,
            f"{name}: {n_elements} elements exceeds the {MAX_TENSOR_ELEMENTS} limit",
        )

    data = spec.get("data")
    _require(isinstance(data, str), f"{name}: data must be a base64 string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"{name}: data is not valid base64: {exc}")

    want_bytes = n_elements * ALLOWED_DTYPES[dtype]
    _require(
        len(raw) == want_bytes,
        f"{name}: got {len(raw)} bytes, shape/dtype require exactly {want_bytes}",
    )

    # `frombuffer` raises on a zero-length buffer, and an empty tensor is a legitimate
    # observation (a training run that took no steps produces one), so it is built
    # directly rather than decoded.
    if n_elements == 0:
        return torch.empty(shape, dtype=getattr(torch, dtype))

    # `frombuffer` reinterprets bytes; it runs no candidate code. The copy is so the
    # tensor does not alias a buffer the caller may reuse, and `reshape` after the fact
    # is safe because the element count was verified above.
    flat = torch.frombuffer(bytearray(raw), dtype=getattr(torch, dtype))
    return flat.reshape(shape).clone()


def _decode_value(value: Any, *, name: str, depth: int = 0):
    """Decode one observation value, recursively for lists."""
    _require(depth <= MAX_TENSOR_RANK, f"{name}: nesting is too deep")

    if isinstance(value, dict):
        _require(
            set(value) == {"__tensor__"},
            f"{name}: objects must be exactly a __tensor__ wrapper",
        )
        return _decode_tensor(value["__tensor__"], name=name)

    if isinstance(value, list):
        _require(
            len(value) <= MAX_LIST_ELEMENTS,
            f"{name}: list of {len(value)} exceeds the {MAX_LIST_ELEMENTS} limit",
        )
        return [_decode_value(v, name=name, depth=depth + 1) for v in value]

    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value

    if isinstance(value, float):
        # NaN and infinity are legitimate observations -- "the loss went non-finite" is
        # a thing a check needs to see -- so they are preserved rather than rejected.
        # They are called out here only because `json.loads` accepts them silently and a
        # later comparison against a threshold would quietly be False for NaN.
        return value

    if isinstance(value, str):
        _require(len(value) <= MAX_DETAIL_LEN, f"{name}: string is too long")
        return value

    raise ProtocolError(f"{name}: unsupported value type {type(value).__name__}")


def decode_observations(text: str, *, marker: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Extract, validate and decode the observation document from container stdout.

    `marker` carries this job's nonce. Candidate code shares stdout with the probe and
    can print anything it likes, so the document is taken from the LAST block bearing the
    secret marker -- a candidate that prints its own block cannot displace the real one,
    and one that prints an unnonced block is not parsed at all.

    Returns `(observations, errors)`. Raises `ProtocolError` for anything malformed;
    callers must treat that as "grading did not happen", never as a failed check.
    """
    _require(
        len(text) <= MAX_PAYLOAD_BYTES,
        f"container stdout is {len(text)} bytes, over the {MAX_PAYLOAD_BYTES} limit",
    )

    at = text.rfind(marker)
    _require(at >= 0, "no authenticated observation block in container output")
    body = text[at + len(marker) :]

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"observation block is not valid JSON: {exc}")

    _require(isinstance(payload, dict), "observation document must be an object")
    unexpected = set(payload) - {"observations", "errors"}
    _require(not unexpected, f"unexpected top-level fields {sorted(unexpected)}")

    raw_obs = payload.get("observations") or {}
    raw_errs = payload.get("errors") or {}
    _require(isinstance(raw_obs, dict), "observations must be an object")
    _require(isinstance(raw_errs, dict), "errors must be an object")
    _require(
        len(raw_obs) + len(raw_errs) <= MAX_OBSERVATIONS,
        f"{len(raw_obs) + len(raw_errs)} entries exceeds the {MAX_OBSERVATIONS} limit",
    )

    for key in list(raw_obs) + list(raw_errs):
        _require(isinstance(key, str), "observation names must be strings")
        _require(
            0 < len(key) <= MAX_NAME_LEN, f"observation name {key[:40]!r} has a bad length"
        )

    observations = {k: _decode_value(v, name=k) for k, v in raw_obs.items()}
    errors = {}
    for k, v in raw_errs.items():
        _require(isinstance(v, str), f"{k}: error detail must be a string")
        errors[k] = v[:MAX_DETAIL_LEN]
    return observations, errors


# --------------------------------------------------------------------------- #
# Candidate-side encoder
# --------------------------------------------------------------------------- #
# This is the exact source the probe uses inside the candidate container. It lives here,
# next to the decoder it must agree with, and is injected as text rather than imported:
# the candidate container has no `trgym` on its filesystem, which is the entire point.
ENCODER_SOURCE = r'''
import base64 as _b64, math as _math

_ALLOWED = {"torch.float32": "float32", "torch.int64": "int64", "torch.bool": "bool"}

def _enc_tensor(t):
    import torch
    t = t.detach().cpu().contiguous()
    name = _ALLOWED.get(str(t.dtype))
    if name is None:
        t = t.to(torch.float32)
        name = "float32"
    return {"__tensor__": {"dtype": name, "shape": list(t.shape),
                           "data": _b64.b64encode(t.numpy().tobytes()).decode("ascii")}}

def _enc(v):
    import torch
    if isinstance(v, torch.Tensor):
        return _enc_tensor(v)
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, int) or isinstance(v, str):
        return v
    if isinstance(v, float):
        return v
    return str(v)
'''


def encode_tensor(tensor) -> dict:
    """Host-side mirror of `_enc_tensor`, so tests can round-trip without a container."""
    import torch

    tensor = tensor.detach().cpu().contiguous()
    name = {"torch.float32": "float32", "torch.int64": "int64",
            "torch.bool": "bool"}.get(str(tensor.dtype))
    if name is None:
        tensor = tensor.to(torch.float32)
        name = "float32"
    return {
        "__tensor__": {
            "dtype": name,
            "shape": list(tensor.shape),
            "data": base64.b64encode(tensor.numpy().tobytes()).decode("ascii"),
        }
    }


__all__ = [
    "ALLOWED_DTYPES",
    "ENCODER_SOURCE",
    "MAX_OBSERVATIONS",
    "MAX_PAYLOAD_BYTES",
    "MAX_TENSOR_ELEMENTS",
    "ProtocolError",
    "decode_observations",
    "encode_tensor",
]
