import pytest

from forecast.tokens import project


def test_quadratic_total_prompt():
    # K·L1 + g·K(K−1)/2
    s = project(steps=10, start_prompt=80_000, growth=3_000, output_per_step=1_000, cache_hit=0.0)
    assert s.input == pytest.approx(10 * 80_000 + 3_000 * 10 * 9 / 2)
    assert s.cache_read == 0 and s.cache_write == 0
    assert s.output == pytest.approx(10_000)


def test_cold_cache_two_steps_matches_hand_count():
    # call 1: prompt 80k, nothing cached, writes 80k
    # call 2: prompt 83k, prefix 80k (72k read, 8k uncached), writes 3k
    s = project(steps=2, start_prompt=80_000, growth=3_000, output_per_step=500, cache_hit=0.9)
    assert s.cache_write == pytest.approx(83_000)
    assert s.cache_read == pytest.approx(72_000)
    assert s.input == pytest.approx(8_000)
    assert s.output == pytest.approx(1_000)
    assert s.total == pytest.approx(163_000 + 1_000)


def test_warm_cache_first_call_reads_instead_of_writing():
    cold = project(1, 80_000, 3_000, 0, 1.0, warm_cache=False)
    warm = project(1, 80_000, 3_000, 0, 1.0, warm_cache=True)
    assert cold.cache_write == pytest.approx(80_000) and cold.cache_read == 0
    assert warm.cache_read == pytest.approx(80_000) and warm.cache_write == 0


def test_split_always_sums_to_total_prompt_plus_output():
    for k in (0.5, 1, 1.5, 2, 7.3, 60):
        s = project(k, 70_000, 2_500, 800, 0.85)
        expected = k * 70_000 + 2_500 * k * (k - 1) / 2 + k * 800
        assert s.total == pytest.approx(expected)
        assert min(s.input, s.cache_read, s.cache_write, s.output) >= 0


def test_zero_steps():
    assert project(0, 80_000, 3_000, 1_000, 0.9).total == 0


def test_monotone_in_steps():
    totals = [project(k, 80_000, 3_000, 1_000, 0.9).total for k in (1, 2, 4, 8, 20, 50)]
    assert totals == sorted(totals)
