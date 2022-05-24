import pytest

import numpy as np
from math import isclose
from scipy.stats import multivariate_normal

from py21cmmc.cosmoHammer import Params
from py21cmmc.prior import DataWhitener, PriorBase, PriorFunction, PriorGaussianKDE


@pytest.fixture(scope="module")
def astro_params():
    ap = {
        "L_X": [40.0, 38.0, 42.0, 0.05],
        "NU_X_THRESH": [500.0, 200.0, 1500.0, 20.0],
    }
    return Params(*[(k, v) for k, v in ap.items()])


@pytest.fixture(scope="module")
def chain(astro_params):
    mean = np.array([ap[0] for ap in astro_params]).reshape(1, -1)
    std = np.array([ap[-1] for ap in astro_params]).reshape(1, -1)

    sample = np.random.normal(size=(100, len(astro_params.keys)))

    return mean + sample * std


@pytest.fixture(scope="module")
def f(astro_params):
    mean = np.array([ap[0] for ap in astro_params])
    cov = [[1.0, 0.0], [0.0, 1.0]]
    return multivariate_normal(mean, cov).logpdf


def test_prior_base():
    prior = PriorBase()
    assert prior.computePrior == prior.computeLikelihood
    with pytest.raises(NotImplementedError):
        prior.computePrior([0.0, 0.0])


def test_prior_function(f, astro_params):
    prior = PriorFunction(arg_names=astro_params.keys)
    with pytest.raises(ValueError):
        prior.computePrior([0.0, 0.0])

    prior = PriorFunction(arg_names=astro_params.keys, f=f)
    prior.computePrior([0.0, 0.0])


def test_prior_gaussian_kde(chain, astro_params, f):
    prior = PriorGaussianKDE(chain=chain, arg_names=astro_params.keys)
    prior.computePrior([0.0, 0.0])

    mean = np.array([ap[0] for ap in astro_params]).reshape(1, -1)
    prior_kde = PriorGaussianKDE(
        chain=mean, arg_names=astro_params.keys, bandwidth=1.0, whiten=False
    )
    prior_function = PriorFunction(arg_names=astro_params.keys, f=f)

    assert isclose(
        prior_kde.computePrior([0.0, 0.0]), prior_function.computePrior([0.0, 0.0])
    )


def test_whitener(chain):
    algorithms = [None, "rescale", "PCA", "ZCA"]
    dw = {}
    for algo in algorithms:
        dw[algo] = DataWhitener(algo)
        dw[algo].fit(chain)

    with pytest.raises(ValueError):
        DataWhitener("bla").fit(chain)

    samples = [
        np.array([0.0, 0.0]),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    ]
    for sample in samples:
        whiten_unwhiten_res = [
            dw[algo].unwhiten(dw[algo].whiten(sample)) for algo in algorithms
        ]
        for res in whiten_unwhiten_res:
            assert sample.shape == res.shape
            assert np.allclose(whiten_unwhiten_res[0], res)
