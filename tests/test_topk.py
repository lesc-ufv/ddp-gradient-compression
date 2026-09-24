import torch

from ddp_gradient_compression.hooks.topk import (
    pack_sparse_payload,
    select_topk_with_error_feedback,
    unpack_sparse_payload,
)


def test_topk_selects_largest_magnitudes_and_updates_residual():
    gradient = torch.tensor([0.1, -5.0, 2.0, 0.25], dtype=torch.float32)
    residual = torch.zeros_like(gradient)

    indices32, values, k = select_topk_with_error_feedback(
        gradient=gradient,
        residual=residual,
        keep_ratio=0.5,
        use_error_feedback=True,
    )

    assert k == 2

    selected = set(indices32.tolist())
    assert selected == {1, 2}

    selected_values = {
        int(i): float(v)
        for i, v in zip(indices32.tolist(), values.tolist())
    }
    assert selected_values[1] == -5.0
    assert selected_values[2] == 2.0

    torch.testing.assert_close(
        residual,
        torch.tensor([0.1, 0.0, 0.0, 0.25], dtype=torch.float32),
    )


def test_disabling_error_feedback_zeros_residual():
    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    residual = torch.ones_like(gradient)

    select_topk_with_error_feedback(
        gradient=gradient,
        residual=residual,
        keep_ratio=0.5,
        use_error_feedback=False,
    )

    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_sparse_payload_roundtrip_preserves_int32_bits_exactly():
    indices = torch.tensor(
        [0, 1, 17, 123456, torch.iinfo(torch.int32).max - 1],
        dtype=torch.int32,
    )
    values = torch.tensor(
        [1.0, -2.0, 3.5, 0.125, -7.25],
        dtype=torch.float32,
    )

    payload = pack_sparse_payload(indices, values)
    decoded_indices, decoded_values = unpack_sparse_payload(payload)

    assert torch.equal(decoded_indices, indices)
    torch.testing.assert_close(decoded_values, values)


def test_topk_uses_at_least_one_value():
    gradient = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    residual = torch.zeros_like(gradient)

    _, _, k = select_topk_with_error_feedback(
        gradient=gradient,
        residual=residual,
        keep_ratio=0.000001,
        use_error_feedback=True,
    )

    assert k == 1
