import numpy as np
import pandas as pd
import pytest

from tsorb.ElectricalLoadProfile import ElectricalLoadProfile
from tsorb.utils.InputData import DataExchangeCsv


def generator(seed=7, **kwargs):
    np.random.seed(seed)
    settings = dict(get_hot_water=True, freq="15min", resample_mean=True)
    settings.update(kwargs)
    return ElectricalLoadProfile(DataExchangeCsv(), 3, **settings)


def test_week_at_quarter_hours():
    profiles = generator().get_period_profiles("2025-01-15", "2025-01-21")

    assert len(profiles) == 7 * 96
    assert profiles.index[0] == pd.Timestamp("2025-01-15 00:00")
    assert profiles.index[-1] == pd.Timestamp("2025-01-21 23:45")
    assert set(profiles.columns) == {"Load", "AppHeatGain", "OccActive", "OccNotActive", "HotWater"}
    assert not profiles.isna().any().any()
    assert (profiles["Load"] > 0).all()


def test_day_types_and_numbers_follow_the_calendar():
    elp = generator()
    calls = []
    original = elp.run

    def spy(year, day_of_week, day_in_year):
        calls.append((year, day_of_week, day_in_year))
        original(year, day_of_week, day_in_year)

    elp.run = spy
    elp.run_for_period("2025-01-17", "2025-01-20")

    assert calls == [(2025, "wd", 17), (2025, "we", 18), (2025, "we", 19), (2025, "wd", 20)]


def test_a_second_period_is_simulated_anew():
    elp = generator()

    assert len(elp.get_period_profiles("2025-03-03", "2025-03-04")) == 2 * 96
    assert len(elp.get_period_profiles("2025-03-03", "2025-03-05")) == 3 * 96


def test_repeated_request_returns_the_same_realization():
    elp = generator()
    first = elp.get_period_profiles("2025-03-03", "2025-03-04")
    second = elp.get_period_profiles("2025-03-03", "2025-03-04")

    pd.testing.assert_frame_equal(first, second)


def test_same_seed_same_profile():
    one = generator(seed=11).get_period_profiles("2025-01-15", "2025-01-16")
    other = generator(seed=11).get_period_profiles("2025-01-15", "2025-01-16")

    pd.testing.assert_frame_equal(one, other)


def test_mean_resampling_conserves_hot_water_energy():
    elp = generator()
    profiles = elp.get_period_profiles("2025-01-15", "2025-01-21")

    assert profiles["HotWater"].sum() / 4 == pytest.approx(elp.hotWater.sum() / 60)


@pytest.mark.parametrize(
    "start, end",
    [("2025-01-21", "2025-01-15"), ("2024-12-31", "2025-01-01")],
)
def test_invalid_periods_are_rejected(start, end):
    with pytest.raises(ValueError):
        generator().run_for_period(start, end)
