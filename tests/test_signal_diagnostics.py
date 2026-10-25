"""Readout diagnostics must not fit normalizers or labels across date boundaries."""
import numpy as np
import pytest


def module():
    from rlasset import signal_diagnostics
    return signal_diagnostics


def test_five_day_labels_end_inside_partition_and_match_manual_returns():
    # A boundary crossing would turn training labels into future information.
    log_prices = np.arange(80, dtype=float)[:, None] * np.array([[.01, .03]])
    prices = np.exp(log_prices)
    features = np.c_[np.arange(80), np.arange(80) ** 2]
    samples = module().make_samples(prices, features, 20, 34)
    assert samples['decision_indices'].tolist() == [20, 25]
    assert samples['label_end_indices'].tolist() == [25, 30]
    np.testing.assert_allclose(samples['y'][:, 0], .1)
    np.testing.assert_allclose(samples['y'][:, 1], 0., atol=1e-12)
    np.testing.assert_array_equal(samples['x'], features[[20, 25]])


def test_readout_standardization_uses_only_training_and_predicts_constant_target():
    # Full-sample fitting or including future rows shifts this fitted intercept.
    train_x = np.c_[np.arange(30), np.ones(30)]
    y = np.tile([.02, .15], (30, 1))
    fitted = module().fit_readout(train_x, y)
    np.testing.assert_allclose(fitted['mean'], [14.5, 1])
    prediction = module().predict_readout(fitted, np.array([[1e6, -1e6]]))
    np.testing.assert_allclose(prediction, [[.02, .15]], atol=1e-10)
    np.testing.assert_allclose(fitted['coef'], 0., atol=1e-10)


def test_future_data_perturbation_leaves_training_fit_identical():
    rng = np.random.default_rng(4)
    prices = np.exp(np.cumsum(rng.normal(0, .01, (100, 3)), axis=0))
    features = rng.normal(size=(100, 8))
    left = module().make_samples(prices, features, 20, 69)
    prices[70:] *= 100
    features[70:] += 9999
    right = module().make_samples(prices, features, 20, 69)
    a = module().fit_readout(left['x'], left['y'])
    b = module().fit_readout(right['x'], right['y'])
    for key in ('mean', 'scale', 'coef', 'intercept', 'target_mean'):
        np.testing.assert_array_equal(a[key], b[key])


def test_metrics_use_training_mean_as_reference_and_keep_negative_skill():
    # Comparing to the evaluation mean would silently fit a benchmark on test.
    metrics = module().readout_metrics(np.array([[1., 3.], [3., 5.]]),
                                       np.zeros((2, 2)), np.array([0., 0.]),
                                       np.array([3., 5.]))
    assert metrics['return']['mse'] == 5.
    assert metrics['return']['skill_vs_train_mean'] == 0.
    assert metrics['volatility']['persistence_mse'] == 0.
    assert metrics['volatility']['skill_vs_persistence'] is None


@pytest.mark.parametrize('case', ['short', 'nonfinite', 'shape'])
def test_invalid_diagnostic_data_fails_explicitly(case):
    prices, features = np.ones((60, 3)), np.ones((60, 4))
    if case == 'nonfinite':
        features[30, 0] = np.nan
    if case == 'shape':
        features = features[:-1]
    with pytest.raises(ValueError):
        module().make_samples(prices, features, 20, 23 if case == 'short' else 59)
