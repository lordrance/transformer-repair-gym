"""The candidate->trusted channel must be safe against the candidate that produced it.

These are host-side and need no Docker: the decoder is pure Python plus `frombuffer`, so
the hostile-input cases can all be expressed directly. That matters because this is the
one place where attacker-controlled bytes enter the trusted process.
"""

from __future__ import annotations

import base64
import json

import pytest
import torch

from trgym.repo.obs_protocol import (
    MAX_TENSOR_ELEMENTS,
    ProtocolError,
    decode_observations,
    encode_tensor,
)

MARKER = "<<<TRGYM_OBS:deadbeef>>>"


def wire(observations: dict | None = None, errors: dict | None = None,
         *, marker: str = MARKER, prefix: str = "") -> str:
    return prefix + marker + json.dumps(
        {"observations": observations or {}, "errors": errors or {}}
    )


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tensor",
    [
        torch.randn(2, 3),
        torch.zeros(0),                       # empty is legal and must survive
        torch.tensor([1, 2, 3], dtype=torch.int64),
        torch.tensor([True, False, True]),
        torch.randn(2, 3, 4, 5),
    ],
)
def test_tensor_round_trips_exactly(tensor: torch.Tensor) -> None:
    obs, _ = decode_observations(wire({"t": encode_tensor(tensor)}), marker=MARKER)
    got = obs["t"]
    assert got.dtype == tensor.dtype
    assert got.shape == tensor.shape
    assert torch.equal(got, tensor)


def test_scalars_and_nested_lists_round_trip() -> None:
    payload = {"n": 7, "x": 1.5, "flag": True, "nothing": None,
               "trace": [0.1, 0.2, 0.3], "name": "ok"}
    obs, _ = decode_observations(wire(payload), marker=MARKER)
    assert obs == payload


def test_non_finite_values_are_preserved_not_rejected() -> None:
    """`loss became NaN` is an observation a check needs, not a malformed document."""
    body = MARKER + '{"observations": {"loss": NaN, "g": Infinity}, "errors": {}}'
    obs, _ = decode_observations(body, marker=MARKER)
    assert obs["loss"] != obs["loss"]          # NaN
    assert obs["g"] == float("inf")


# --------------------------------------------------------------------------- #
# Framing and authentication
# --------------------------------------------------------------------------- #
def test_unmarked_output_is_not_a_document() -> None:
    with pytest.raises(ProtocolError, match="no authenticated observation block"):
        decode_observations('{"observations": {}, "errors": {}}', marker=MARKER)


def test_candidate_cannot_forge_a_block_without_the_nonce() -> None:
    """A candidate printing an unnonced block must not be parsed at all."""
    forged = '<<<TRGYM_OBS>>>{"observations": {"pwned": 1}, "errors": {}}'
    with pytest.raises(ProtocolError):
        decode_observations(forged, marker=MARKER)


def test_last_marked_block_wins() -> None:
    """The probe writes last, so a candidate that prints an earlier block loses."""
    text = wire({"real": 0}, prefix=wire({"forged": 1}))
    obs, _ = decode_observations(text, marker=MARKER)
    assert obs == {"real": 0}


def test_candidate_prefix_noise_is_ignored() -> None:
    obs, _ = decode_observations(wire({"a": 1}, prefix="chatty stdout\n" * 50),
                                 marker=MARKER)
    assert obs == {"a": 1}


# --------------------------------------------------------------------------- #
# Hostile input
# --------------------------------------------------------------------------- #
def test_oversized_payload_is_rejected_before_parsing() -> None:
    from trgym.repo import obs_protocol

    with pytest.raises(ProtocolError, match="over the"):
        decode_observations("x" * (obs_protocol.MAX_PAYLOAD_BYTES + 1), marker=MARKER)


def test_declared_element_count_is_rejected_without_allocating() -> None:
    """A petabyte-shaped tensor must fail on arithmetic, not on an OOM kill."""
    spec = {"__tensor__": {"dtype": "float32", "shape": [10**6, 10**6], "data": ""}}
    with pytest.raises(ProtocolError, match="exceeds the"):
        decode_observations(wire({"bomb": spec}), marker=MARKER)


def test_shape_and_data_length_must_agree() -> None:
    spec = encode_tensor(torch.randn(4, 4))
    spec["__tensor__"]["shape"] = [4, 5]
    with pytest.raises(ProtocolError, match="require exactly"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_truncated_data_is_rejected() -> None:
    spec = encode_tensor(torch.randn(8))
    raw = base64.b64decode(spec["__tensor__"]["data"])
    spec["__tensor__"]["data"] = base64.b64encode(raw[:-4]).decode()
    with pytest.raises(ProtocolError, match="require exactly"):
        decode_observations(wire({"t": spec}), marker=MARKER)


@pytest.mark.parametrize("dtype", ["float64", "complex64", "object", "", None])
def test_dtype_allowlist(dtype) -> None:
    spec = {"__tensor__": {"dtype": dtype, "shape": [1], "data": "AAAAAA=="}}
    with pytest.raises(ProtocolError, match="is not allowed"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_negative_dimension_is_rejected() -> None:
    spec = {"__tensor__": {"dtype": "float32", "shape": [-1], "data": ""}}
    with pytest.raises(ProtocolError, match="non-negative"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_bool_is_not_accepted_as_a_dimension() -> None:
    """`isinstance(True, int)` is True in Python; the shape check must not be fooled."""
    spec = {"__tensor__": {"dtype": "float32", "shape": [True], "data": "AAAAAA=="}}
    with pytest.raises(ProtocolError, match="non-negative"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_rank_limit() -> None:
    spec = {"__tensor__": {"dtype": "bool", "shape": [1] * 12, "data": "AA=="}}
    with pytest.raises(ProtocolError, match="rank"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_invalid_base64_is_rejected() -> None:
    spec = {"__tensor__": {"dtype": "float32", "shape": [1], "data": "not base64!!"}}
    with pytest.raises(ProtocolError, match="base64"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_extra_tensor_fields_are_rejected() -> None:
    spec = encode_tensor(torch.randn(2))
    spec["__tensor__"]["requires_grad"] = True
    with pytest.raises(ProtocolError, match="unexpected tensor fields"):
        decode_observations(wire({"t": spec}), marker=MARKER)


def test_arbitrary_objects_are_not_decodable() -> None:
    """The only object form on the wire is a tensor wrapper; nothing else is honoured."""
    with pytest.raises(ProtocolError, match="__tensor__ wrapper"):
        decode_observations(wire({"t": {"__reduce__": ["os.system", ["echo pwned"]]}}),
                            marker=MARKER)


def test_unexpected_top_level_fields_are_rejected() -> None:
    body = MARKER + json.dumps({"observations": {}, "errors": {}, "verdict": "PASS"})
    with pytest.raises(ProtocolError, match="unexpected top-level"):
        decode_observations(body, marker=MARKER)


def test_too_many_observations_is_rejected() -> None:
    from trgym.repo import obs_protocol

    many = {f"k{i}": 1 for i in range(obs_protocol.MAX_OBSERVATIONS + 1)}
    with pytest.raises(ProtocolError, match="exceeds the"):
        decode_observations(wire(many), marker=MARKER)


def test_malformed_json_is_a_protocol_error_not_a_crash() -> None:
    with pytest.raises(ProtocolError, match="not valid JSON"):
        decode_observations(MARKER + "{not json", marker=MARKER)


def test_error_details_are_truncated() -> None:
    from trgym.repo import obs_protocol

    _, errs = decode_observations(wire(errors={"c": "x" * 99999}), marker=MARKER)
    assert len(errs["c"]) == obs_protocol.MAX_DETAIL_LEN


def test_element_limit_is_enforced_on_a_real_buffer() -> None:
    """The limit must hold for a tensor whose bytes really are present."""
    n = MAX_TENSOR_ELEMENTS + 1
    spec = {"__tensor__": {"dtype": "bool", "shape": [n],
                           "data": base64.b64encode(b"\x00" * n).decode()}}
    with pytest.raises(ProtocolError, match="exceeds the"):
        decode_observations(wire({"t": spec}), marker=MARKER)
