import numpy as np

from mlx_rl.rollout import Rollout, build_training_arrays


def _mk(prompt, completion, lps):
    return Rollout(
        prompt_tokens=prompt,
        completion_tokens=completion,
        sampling_logprobs=lps,
        text="",
    )


def test_build_training_arrays_alignment():
    # prompt [5, 6], completion [7, 8, 9]: seq = [5,6,7,8,9]
    r = _mk([5, 6], [7, 8, 9], [-0.1, -0.2, -0.3])
    inp, tgt, mask, old_lp = build_training_arrays([r], pad_id=0)
    np.testing.assert_array_equal(inp[0], [5, 6, 7, 8])
    np.testing.assert_array_equal(tgt[0], [6, 7, 8, 9])
    # completion targets are positions 1..3 (predicting tokens 7, 8, 9)
    np.testing.assert_array_equal(mask[0], [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(old_lp[0], [0.0, -0.1, -0.2, -0.3])


def test_build_training_arrays_padding_and_mask():
    a = _mk([1, 2], [3], [-0.5])
    b = _mk([1], [2, 3, 4, 5], [-0.1, -0.2, -0.3, -0.4])
    inp, tgt, mask, old_lp = build_training_arrays([a, b], pad_id=99)
    assert inp.shape == tgt.shape == mask.shape == old_lp.shape == (2, 4)
    # short sequence right-padded with pad_id, padding masked out
    np.testing.assert_array_equal(inp[0], [1, 2, 3, 99])
    np.testing.assert_array_equal(mask[0], [0.0, 1.0, 0.0, 0.0])
    # sequence b: all four completion targets live
    np.testing.assert_array_equal(mask[1], [1.0, 1.0, 1.0, 1.0])
    assert mask.sum() == 5.0  # 1 + 4 completion tokens total
