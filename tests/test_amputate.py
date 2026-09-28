import pytest

from fillwise.amputate import amputate
from fillwise.data import load_breast_cancer_complete

DS = load_breast_cancer_complete()
COLS = list(DS.X.columns[:10])


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_realised_rate_close_to_requested(mechanism, rate):
    _, mask = amputate(DS.X, mechanism, rate, columns=COLS, driver="worst_area", seed=1)
    realised = mask[COLS].to_numpy().mean()
    assert abs(realised - rate) < 0.05


def test_mnar_masks_the_tail():
    _, mask = amputate(DS.X, "mnar", 0.3, columns=COLS, seed=3)
    col = COLS[0]
    masked_mean = DS.X[col][mask[col]].mean()
    observed_mean = DS.X[col][~mask[col]].mean()
    assert masked_mean > observed_mean * 1.1


def test_mar_follows_the_driver_not_the_value():
    _, mask = amputate(DS.X, "mar", 0.3, columns=COLS, driver="worst_area", seed=3)
    driver = DS.X["worst_area"]
    hit_any = mask[COLS].any(axis=1)
    assert driver[hit_any].mean() > driver[~hit_any].mean()
    # the driver column itself must stay fully observed
    assert not mask["worst_area"].any()


def test_mcar_is_uncorrelated_with_values():
    _, mask = amputate(DS.X, "mcar", 0.3, columns=COLS, seed=3)
    col = COLS[0]
    masked_mean = DS.X[col][mask[col]].mean()
    observed_mean = DS.X[col][~mask[col]].mean()
    assert abs(masked_mean - observed_mean) / observed_mean < 0.15


def test_seed_reproducibility():
    a1, m1 = amputate(DS.X, "mcar", 0.2, columns=COLS, seed=7)
    a2, m2 = amputate(DS.X, "mcar", 0.2, columns=COLS, seed=7)
    assert m1.equals(m2)
    assert a1.equals(a2)


def test_bad_args_raise():
    with pytest.raises(ValueError):
        amputate(DS.X, "mcar", 1.5)
    with pytest.raises(ValueError):
        amputate(DS.X, "banana", 0.3)
    with pytest.raises(ValueError):
        amputate(DS.X, "mar", 0.3, columns=list(DS.X.columns))  # no driver possible
