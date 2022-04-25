"""Functions to calculate spherical/cilyndrical power spectrum."""
import numpy as np


def ps1D(
    lc,
    cell_size,
    n_psbins=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    calculate_variance=False,
):
    """Calculating 1D PS for a series of redshifts for one lightcone.

    Parameters
    ----------
        lc : array
            Lightcone.
        cell_size : float
            Simulation voxel size (in Mpc).
        n_psbins : int
            Number of PS bins.
        logk : bool
            If bins should be in log or linear space.
        convert_to_delta : bool
            Either to convert from power to non-dimensional delta.
        chunk_skip : int
            In redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin into account.
            If `None`, it amounts to the lightcone sky-plane size.
        calculate_variance : bool
            Either to calculate sample variance of each bin or not.

    Returns
    -------
        PS : dict
            Power spectrum and its sample variance (if flag is turned on) for all redshift bins.
        k_values : array
            Centers of k bins.
    """
    PS, k_values = _power_1D(
        lc,
        cell_size=cell_size,
        n_psbins=n_psbins,
        chunk_skip=chunk_skip,
        logk=logk,
        calculate_variance=calculate_variance,
    )
    if convert_to_delta is True:
        conversion_factor = k_values ** 3 / (2 * np.pi ** 2)
        PS["power"] = PS["power"] * conversion_factor[np.newaxis, ...]
        if calculate_variance:
            PS["var_power"] = PS["var_power"] * conversion_factor[np.newaxis, ...] ** 2

    return PS, k_values


def ps2D(
    lc,
    cell_size,
    n_psbins_par=12,
    n_psbins_perp=12,
    logk=True,
    convert_to_delta=True,
    chunk_skip=None,
    calculate_variance=False,
):
    """Calculating 2D PS for a series of redshifts for one lightcone.

    Parameters
    ----------
        lc : array
            Lightcone.
        redshifts : list
            List of redshifts for which the lightcone has been computed.
        cell_size : float
            Simulation voxel size (in Mpc).
        n_psbins_par : int
            Number of PS bins in LoS direction.
        n_psbins_perp : int
            Number of PS bins in sky-plane direction.
        logk : bool
            If bins should be in log or linear space.
        convert_to_delta : bool
            Either to convert from power to non-dimensional delta.
        chunk_skip : int
            In redshift dimension of the lightcone,
            PS is calculated on chunks `chunk_skip` apart.
            Eg. `chunk_skip = 2` amounts in taking every second redshift bin
            into account. If `None`, it amounts to the lightcone sky-plane size.
        calculate_variance : bool
            Either to calculate sample variance of each bin or not.

    Returns
    -------
        PS : dict
            Power spectrum and its sample variance (if flag is turned on) for all redshift bins.
        k_values_perp : array
            Centers of k_perp bins.
        k_values_par : array
            Centers of k_par bins.
    """
    PS, k_values_perp, k_values_par = _power_2D(
        lc,
        cell_size=cell_size,
        n_psbins_par=n_psbins_par,
        n_psbins_perp=n_psbins_perp,
        chunk_skip=chunk_skip,
        logk=logk,
        calculate_variance=calculate_variance,
    )
    if convert_to_delta is True:
        k_values_cube = np.meshgrid(
            k_values_par, k_values_perp
        )  # all k_values on the 2D grid
        conversion_factor = (k_values_cube[1] ** 2 * k_values_cube[0]) / (
            4 * np.pi ** 2
        )  # pre-factor k_perp**2 * k_par
        PS["power"] = PS["power"] * conversion_factor[np.newaxis, ...]
        if calculate_variance:
            PS["var_power"] = PS["var_power"] * conversion_factor[np.newaxis, ...] ** 2

    return PS, k_values_perp, k_values_par


def _power_1D(
    lightcone,
    cell_size,
    n_psbins,
    chunk_skip,
    logk,
    calculate_variance,
):
    HII_DIM = lightcone.shape[0]
    n_slices = lightcone.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k

    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    # maximal mode will be k_max * sqrt(3)
    if logk:
        k_bins = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(3) * k_max + epsilon),
            n_psbins + 1,
        )
    else:
        k_bins = np.linspace(
            k_min - epsilon, np.sqrt(3) * k_max + epsilon, n_psbins + 1
        )
    # grid of all k_values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_sphere = np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2 + k_cube[2] ** 2)
    # return a bin index across flattened k_sphere array
    k_sphere_digits = np.digitize(k_sphere.flatten(), k_bins)
    # count occurence of modes in each bin & cut out all values outside the edges
    k_binsum = np.bincount(k_sphere_digits, minlength=n_psbins + 2)[1:-1]
    # geometrical means for values
    k_values = np.sqrt(k_bins[:-1] * k_bins[1:])

    lightcones = []  # all chunks that need to be computed

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        lightcones.append(lightcone[..., start:end])

    lightcones = np.array(lightcones, dtype=np.float32)

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size ** 3

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V
        # calculating average power as a bin count with PS as weights
        res = {}
        res["power"] = (
            np.bincount(
                k_sphere_digits, weights=PS_box.flatten(), minlength=n_psbins + 2
            )[1:-1]
            / k_binsum
        )
        # calculating average square of the power, used for estimating sample variance
        if calculate_variance:
            p_sq = (
                np.bincount(
                    k_sphere_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins + 2,
                )[1:-1]
                / k_binsum
            )
            res["var_power"] = p_sq - res["power"] ** 2

        return res

    res = [_power(lc) for lc in lightcones]

    P = {key: [] for key in res[0].keys()}
    for r in res:
        for key, value in r.items():
            P[key].append(value)
    P = {key: np.array(value, dtype=np.float32) for key, value in P.items()}

    return P, k_values


def _power_2D(
    lightcone,
    cell_size,
    n_psbins_par,
    n_psbins_perp,
    chunk_skip,
    logk,
    calculate_variance,
):
    HII_DIM = lightcone.shape[0]
    n_slices = lightcone.shape[-1]
    chunk_skip = HII_DIM if chunk_skip is None else chunk_skip
    chunk_indices = list(range(0, n_slices + 1 - HII_DIM, chunk_skip))
    epsilon = 1e-12

    # DFT frequency modes
    k = np.fft.fftfreq(HII_DIM, d=cell_size)
    k = 2 * np.pi * k
    # ignoring 0 and negative modes
    k_min, k_max = k[1], np.abs(k).max()
    if logk:
        # maximal perp mode will be k_max * sqrt(2)
        k_bins_perp = np.logspace(
            np.log10(k_min - epsilon),
            np.log10(np.sqrt(2.0) * k_max + epsilon),
            n_psbins_perp + 1,
        )
        # maximal par mode will be k_max
        k_bins_par = np.logspace(
            np.log10(k_min - epsilon), np.log10(k_max + epsilon), n_psbins_par + 1
        )
    else:
        k_bins_perp = np.linspace(
            k_min - epsilon,
            np.sqrt(2.0) * k_max + epsilon,
            n_psbins_perp + 1,
        )
        k_bins_par = np.linspace(k_min - epsilon, k_max + epsilon, n_psbins_par + 1)

    # grid of all k_values, where k_cube[0], k_cube[1] are perp values, and k_cube[2] par values
    k_cube = np.meshgrid(k, k, k)
    # calculating k_perp, k_par in cylindrical coordinates
    k_cylinder = [np.sqrt(k_cube[0] ** 2 + k_cube[1] ** 2), np.abs(k_cube[2])]
    # return a bin index across flattened k_cylinder, for perp and par
    k_perp_digits = np.digitize(k_cylinder[0].flatten(), k_bins_perp)
    k_par_digits = np.digitize(k_cylinder[1].flatten(), k_bins_par)
    # construct a unique digit counter for a 2D PS array
    # for first k_perp uses range [1, n_psbins_par]
    # for second k_perp uses range [n_psbins_par + 1, 2 * n_psbins_par] etc.
    k_cylinder_digits = (k_perp_digits - 1) * n_psbins_par + k_par_digits
    # now cut out outsiders: zeros, n_psbins_par + 1, n_psbins_perp + 1
    k_cylinder_digits = np.where(
        np.logical_or(k_perp_digits == 0, k_par_digits == 0), 0, k_cylinder_digits
    )
    k_cylinder_digits = np.where(
        np.logical_or(
            k_perp_digits == n_psbins_perp + 1, k_par_digits == n_psbins_par + 1
        ),
        n_psbins_perp * n_psbins_par + 1,
        k_cylinder_digits,
    )
    k_binsum = np.bincount(
        k_cylinder_digits, minlength=n_psbins_par * n_psbins_perp + 2
    )[1:-1]
    # geometrical means for values
    k_values_perp = np.sqrt(k_bins_perp[:-1] * k_bins_perp[1:])
    k_values_par = np.sqrt(k_bins_par[:-1] * k_bins_par[1:])

    lightcones = []  # all chunks that need to be computed

    # appending all chunks together
    for i in chunk_indices:
        start = i
        end = i + HII_DIM
        lightcones.append(lightcone[..., start:end])

    lightcones = np.array(lightcones, dtype=np.float32)

    V = (HII_DIM * cell_size) ** 3
    dV = cell_size ** 3

    def _power(box):
        FT = np.fft.fftn(box) * dV
        PS_box = np.real(FT * np.conj(FT)) / V

        res = {}
        # calculating average power as a bin count with PS as weights
        res["power"] = (
            np.bincount(
                k_cylinder_digits,
                weights=PS_box.flatten(),
                minlength=n_psbins_par * n_psbins_perp + 2,
            )[1:-1]
            / k_binsum
        ).reshape(n_psbins_perp, n_psbins_par)
        if calculate_variance:
            # calculating average square of the power, used for estimating sample variance
            p_sq = (
                np.bincount(
                    k_cylinder_digits,
                    weights=PS_box.flatten() ** 2,
                    minlength=n_psbins_par * n_psbins_perp + 2,
                )[1:-1]
                / k_binsum
            ).reshape(n_psbins_perp, n_psbins_par)
            res["var_power"] = p_sq - res["power"] ** 2

        return res

    res = [_power(lc) for lc in lightcones]

    P = {key: [] for key in res[0].keys()}
    for r in res:
        for key, value in r.items():
            P[key].append(value)
    P = {key: np.array(value, dtype=np.float32) for key, value in P.items()}

    return P, k_values_perp, k_values_par
