"""Module containing additional priors to be used in 21CMMC."""

import logging
import numpy as np
from functools import partial
from sklearn.neighbors import KernelDensity

from . import likelihood

logger = logging.getLogger("21cmFAST")


class PriorBase(likelihood.LikelihoodBase):
    """Base prior class."""

    def __init__(self):
        super().__init__()

        self.computeLikelihood = self.computePrior

    def computePrior(self, arg_values):
        """
        Calculate the log-prior of the instance data given the model.

        Parameters
        ----------
        arg_values : dict
            A list containing all model-dependent quantities required to calculate
            the prior. Explicitly, matches the output of :meth:`~reduce_data`.

        Returns
        -------
        lnP : float
            Logarithm of the prior.
        """
        raise NotImplementedError("The Base prior should never be used directly!")


class PriorFunction(PriorBase):
    """Wrapper class for an arbitrary prior function.

    Parameters
    ----------
    arg_names : list
        List of argument names to be extracted from the context and from which
        `arg_values` list is constructed.
    f : callable
        log-prior function. It is assumed to recieve a list of `arg_values` as
        a parameter, i.e. it is called as `f(arg_values)`.

    """

    def __init__(self, arg_names, f=None):
        super().__init__()
        self.arg_names = arg_names
        self.f = f

    def computePrior(self, arg_values):
        """Calling the prior function."""
        if self.f is None:
            raise ValueError("Prior function is not defined.")
        else:
            return self.f(arg_values)

    def reduce_data(self, ctx):
        """Extracting argument values from the context."""
        params = ctx.getParams()
        arg_values = [v for k, v in params.items() if k in self.arg_names]
        return arg_values


class PriorGaussianKDE(PriorFunction):
    """Gaussian Kernel Density Estimation prior function.

    Given the list of `arg_names` and a chain data to fit KDE with,
    it returns a KDE approximation of the underlying prior.

    Parameters
    ----------
    chain : array
        Array containing data to be fit by KDE, of shape `(n_samples, n_dim)`.
    arg_names : list
        List of argument names, in the same order as in `chain`. Number of arguments
        should be equal to `n_dim`.
    bandwidth : float, optional
        Gaussian bandwidth to use, by default calculates optimal bandwidth.
    whiten : bool, optional
        Either to use whitener with KDE fit or not. Defaults to `True`.
    whitening_algorithm : str, optional
        Whitening algorithm to use, one of one of `["PCA", "ZCA", "rescale"]`,
        defaults to "rescale".
    """

    def __init__(
        self,
        chain,
        arg_names,
        bandwidth=None,
        whiten=True,
        whitening_algorithm="rescale",
    ):
        super().__init__(arg_names)

        if bandwidth is None:
            n_samples, n_dim = chain.shape
            bandwidth = 10 ** self.log_scott(n_samples, n_dim)

        if whiten is False:
            whitening_algorithm = None
        self.dw = DataWhitener(algorithm=whitening_algorithm)
        self.dw.fit(chain)

        self.kde = KernelDensity(bandwidth=bandwidth)
        self.kde.fit(self.dw.whiten(chain))

    def computePrior(self, arg_values):
        """Computing KDE prior.

        Firstly extracting argument values, followed by lnP calculation
        from KDE function, with possible whitening step.
        """
        arg_values = np.array(arg_values).reshape(1, -1)
        arg_values = self.dw.whiten(arg_values)
        lnP = np.squeeze(self.kde.score_samples(arg_values))
        return lnP

    @staticmethod
    def log_scott(n_samples, n_dim):
        """Optimal bandwidth, i.e. Scott's parameter."""
        return -1 / (n_dim + 4) * np.log10(n_samples)


class DataWhitener:
    """Whitening of the data.

    Implements several algorithms, depending on the desired whitening properties.

    Parameters
    ----------
    algorithm : str
        One of `[None, "PCA", "ZCA", "rescale"]`.
        `None`: does nothing.
        "PCA": data is transformed into its PCA space and divided by
        the standard deviation of each dimension
        "ZCA": equivalent to the "PCA", with additional step of rotating
        back to original space. In this case, the final data still
        outputs 'in the same direction'.
        "rescale": calculates mean and standard deviation in each dimension
        and rescales it to zero-mean, unit-variance. In the absence
        of high correlations between dimensions, this is often sufficient.
    """

    def __init__(self, algorithm="rescale"):
        self.algorithm = algorithm

    def fit(self, X, save_data=False):
        """Fitting the whitener on the data X.

        Parameters
        ----------
        X : array
            Of shape `(n_samples, n_dim)`.
        save_data : bool
            If `True`, saves the data and whitened data as `self.data`, `self.whitened_data`.

        Returns
        -------
        X_whiten : array
            Whitened array.
        """
        if self.algorithm is not None:
            self.μ = np.mean(X, axis=0, dtype=np.float128).astype(np.float32)
            Σ = np.cov(X.T)
            evals, evecs = np.linalg.eigh(Σ)

            if self.algorithm == "PCA":
                self.W = np.einsum("ij,kj->ik", np.diag(evals ** (-1 / 2)), evecs)
                self.WI = np.einsum("ij,jk->ik", evecs, np.diag(evals ** (1 / 2)))
            elif self.algorithm == "ZCA":
                self.W = np.einsum(
                    "ij,jk,lk->il", evecs, np.diag(evals ** (-1 / 2)), evecs
                )
                self.WI = np.einsum(
                    "ij,jk,lk->il", evecs, np.diag(evals ** (1 / 2)), evecs
                )
            elif self.algorithm == "rescale":
                self.W = np.identity(len(Σ)) * np.diag(Σ) ** (-1 / 2)
                self.WI = np.identity(len(Σ)) * np.diag(Σ) ** (1 / 2)
            else:
                raise ValueError(
                    "`algorithm` should be either `None`, PCA, ZCA or rescale."
                )

        if save_data:
            self.data = X
            self.whitened_data = self.whiten(X)

    def whiten(self, X):
        """Whiten the data by making it unit covariance.

        Parameters
        ----------
        X : array
            Data to whiten, of shape `(n_samples, n_dims)`.
            `n_dims` has to be the same as self.data.

        Returns
        -------
        whitened_data : array
            Whitened data, of shape `(n_samples, n_dims)`.
        """
        if self.algorithm is None:
            return X

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
            squeeze = True
        else:
            squeeze = False
        X_whitened = np.einsum("ij,kj->ki", self.W, X - self.μ)
        return np.squeeze(X_whitened) if squeeze else X_whitened

    def unwhiten(self, X_whitened):
        """Un-whiten the sample with whitening parameters from the data.

        Parameters
        ----------
        X_whitened : array
            Sample of the data to un-whiten, of shape `(n_samples, n_dims)`.
            `n_dims` has to be the same as `self.data`.

        Returns
        -------
            X : array
                Whitened data, of shape `(n_samples, n_dims)`.
        """
        if self.algorithm is None:
            return X_whitened

        if len(X_whitened.shape) == 1:
            X_whitened = np.expand_dims(X_whitened, axis=0)
            squeeze = True
        else:
            squeeze = False
        X = np.einsum("ij,kj->ki", self.WI, X_whitened) + self.μ
        return np.squeeze(X) if squeeze else X
