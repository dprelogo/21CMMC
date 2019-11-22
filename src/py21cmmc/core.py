"""
A module providing Core Modules for cosmoHammer. This is the basis of the plugin system for py21cmmc.
"""
import copy
import inspect
import logging
import warnings
import numpy as np
from collections import Iterable

import py21cmfast as p21

logger = logging.getLogger("21cmFAST")


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


class NotSetupError(AttributeError):
    def __init__(self):
        default_message = (
            "setup() must have been called on the chain to use this method/attribute!"
        )
        super().__init__(default_message)


class NotAChain(AttributeError):
    def __init__(self):
        default_message = "this Core or Likelihood must be part of a LikelihoodComputationChain to enable this method/attribute!"
        super().__init__(default_message)


class AlreadySetupError(Exception):
    pass


class ModuleBase:
    extra_defining_attributes = (
        []
    )  # extra attributes (in addition to those passed to init) that define equality
    ignore_attributes = (
        []
    )  # attributes to ignore (from those passed to init) for determining equality

    required_cores = []  # Specifies required cores that need to be loaded if this core
    # is loaded. Tuples in the list indicate "or" relationship.

    def __init__(self):
        self._is_setup = False

    def _check_required_cores(self):
        for rc in self.required_cores:
            # Ensure the required_core is a tuple -- we check that at least *one*
            # of the cores in the tuple is in the _cores.
            if not hasattr(rc, "__len__"):
                rc = (rc,)

            if not any([any([isinstance(m, r) for r in rc]) for m in self._cores]):
                raise ValueError(
                    "%s needs the %s to be loaded."
                    % (self.__class__.__name__, rc.__class__.__name__)
                )

    @property
    def chain(self):
        """
        A reference to the LikelihoodComputationChain of which this Core is a part.
        """
        try:
            return self._LikelihoodComputationChain
        except AttributeError:
            raise NotAChain

    @property
    def parameter_names(self):
        return getattr(self.chain.params, "keys", [])

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False

        args = []
        for cls in self.__class__.mro():
            args += inspect.getfullargspec(cls.__init__).args

        args = list(set(args))

        for arg in args + self.extra_defining_attributes:

            if arg == "self" or arg in self.ignore_attributes:
                continue

            try:
                if hasattr(self, "_" + arg):
                    if getattr(self, "_" + arg) != getattr(other, "_" + arg):
                        return False
                elif hasattr(self, arg):
                    if getattr(self, arg) != getattr(other, arg):
                        return False
                else:
                    logger.warning(
                        "parameter {arg} not found in instance".format(arg=arg)
                    )

            except ValueError:
                logger.warning(
                    "parameter {arg} has type which does not allow for comparison".format(
                        arg=arg
                    )
                )

        return True

    @property
    def _cores(self):
        """List of all loaded cores"""
        return self.chain.getCoreModules()

    @property
    def _rq_cores(self):
        """List of all loaded cores that are in the requirements, in order of the requirements"""
        req = flatten(self.required_cores)
        return tuple([core for core in self._cores for r in req if isinstance(core, r)])

    @property
    def core_primary(self):
        """The first core that appears in the requirements"""
        return self._rq_cores[0] if self._rq_cores else self._cores[0]

    def setup(self):
        self._check_required_cores()


class CoreBase(ModuleBase):
    def __init__(self, store=None):
        self.store = store or {}

    def _check_required_cores(self):
        for core in self._cores:
            for rc in self.required_cores:
                if core.__class__.__name__ == rc.__class__.__name__:
                    break
                if core.__class__.__name__ == self.__class__.__name__:
                    raise ValueError(
                        "{this} requires {that} to be loaded.".format(
                            this=self.__class__.__name__, that=rc.__class__.__name__
                        )
                    )

    def prepare_storage(self, ctx, storage):
        """Add variables to special dict which cosmoHammer will automatically store with the chain."""
        for name, storage_function in self.store.items():
            try:
                storage[name] = storage_function(ctx)
            except Exception:
                logger.error(
                    "Exception while trying to evaluate storage function %s" % name
                )
                raise

    def build_model_data(self, ctx):
        """
        Passed a standard context object, should construct model data and place it in the context.

        The data generated by this method should ideally be *deterministic*, so that input parameters (which are
        inherently contained in the `ctx` object) map uniquely to output data. The addition of stochasticity in order
        to produce mock data is done in the :meth:`~convert_model_to_mock` method. All data necessary to full evaulate
        probabilities of mock data from the model data should be determined in this method (including model
        uncertainties, if applicable).

        Parameters
        ----------
        ctx : dict-like
            The context, from which parameters are accessed.

        Returns
        -------
        dct : dict
            A dictionary of data which was simulated.
        """
        pass

    def convert_model_to_mock(self, ctx):
        """
        Given a context object containing data from :meth:`~build_model_data`, generate random mock data, which should
        represent an exact forward-model of the process under investigation.

        Parameters
        ----------
        ctx : dict-like
            The context, from which parameters and other simulated model data can be accessed.
        """
        pass

    def simulate_mock(self, ctx):
        """Generate all mock data and add it to the context."""
        self.build_model_data(ctx)
        self.convert_model_to_mock(ctx)

    def __call__(self, ctx):
        """
        Call the class. By default, it will just build model data, with no stochasticity.
        """
        self.build_model_data(ctx)


class CoreCoevalModule(CoreBase):
    """
    A Core Module which evaluates coeval cubes at given redshift.

    On each iteration, this module will add to the context:

    1. ``init``: an :class:`~py21cmmc._21cmfast.wrapper.InitialConditions` instance
    2. ``perturb``: a :class:`~py21cmmc._21cmfast.wrapper.PerturbedField` instance
    3. ``xHI``: an :class:`~py21cmmc._21cmfast.wrapper.IonizedBox` instance
    4. ``brightness_temp``: a :class:`~py21cmmc._21cmfast.wrapper.BrightnessTemp` instance
    """

    ignore_attributes = ["keep_data_in_memory"]

    def __init__(
        self,
        redshift,
        user_params=None,
        flag_options=None,
        astro_params=None,
        cosmo_params=None,
        regenerate=True,
        z_step_factor=1.02,
        z_heat_max=None,
        change_seed_every_iter=False,
        ctx_variables=None,
        initial_conditions_seed=None,
        **io_options,
    ):
        """
        Initialize the class.

        .. note:: None of the parameters provided here affect the *MCMC* as such; they merely provide a background
                  model on which the MCMC will be performed. Thus for example, passing `HII_EFF_FACTOR=30` in
                  `astro_params` here will be over-written per-iteration if `HII_EFF_FACTOR` is also passed as a
                  `parameter` to an MCMC routine using this Core Module.

        Parameters
        ----------
        redshift : float or array_like
             The redshift(s) at which to evaluate the coeval cubes.
        user_params : dict or :class:`~py21cmmc._21cmfast.wrapper.UserParams`
            Parameters affecting the overall dimensions of the cubes (see :class:`~py21cmmc._21cmfast.wrapper.UserParams`
            for details).
        flag_options : dict or :class:`~py21cmmc._21cmfast.wrapper.FlagOptions`
            Options affecting choices for how the reionization is calculated (see
            :class:`~py21cmmc._21cmfast.wrapper.FlagOptions` for details).
        astro_params : dict or :class:`~py21cmmc._21cmfast.wrapper.AstroParams`
            Astrophysical parameters of reionization (see :class:`~py21cmmc._21cmfast.wrapper.AstroParams` for details).
        cosmo_params : dict or :class:`~py21cmmc._21cmfast.wrapper.CosmoParams`
            Cosmological parameters of the simulations (see :class:`~py21cmmc._21cmfast.wrapper.CosmoParams` for
            details).
        regenerate : bool, optional
            Whether to force regeneration of simulations, even if matching cached data is found.
        do_spin_temp: bool, optional
            Whether to use spin temperature in the calculation, or assume the saturated limit.
        z_step_factor: float, optional
            How large the logarithmic steps between redshift are (if required).
        z_heat_max: float, optional
            Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
            are required to specify the ionization field. Beyond this, the ionization field is specified directly from
            the perturbed density field.
        ctx_variables : list of str, optional
            A list of strings, any number of the following: "brightness_temp", "init", "perturb", "xHI". These each
            correspond to an OutputStruct which will be stored in the context on every iteration. Omitting as many as
            possible is useful in that it reduces the memory that needs to be transmitted to each process. Furthermore,
            in-built pickling has a restriction that arrays cannot be larger than 4GiB, which can be easily over-run
            when passing the hires array in the "init" structure.
        initial_conditions_seed : int, optional
            If not `change_seeds_every_iter`, then this will define the random seed on which the initial conditions
            for _all_ iterations is based. By default, a seed will be chosen at random, _unless_ initial conditions
            exist in cache that match the parameters of this instance (and `regenerate` is False). In this case, the
            seed of the existing box will be adopted.

        Other Parameters
        ----------------
        store :  dict, optional
            The (derived) quantities/blobs to store in the MCMC chain, default empty. See Notes below for details.
        cache_dir : str, optional
            The directory in which to search for the boxes and write them. By default, this is the directory given by
            ``boxdir`` in the configuration file, ``~/.21CMMC/config.yml``. Note that for *reading* data, while the
            specified `direc` is searched first, the default directory will *also* be searched if no appropriate data is
            found in `direc`.
        cache_ionize : bool, optional
            Whether to cache ionization data sets (done before parameter retention step). Default False.


        Notes
        -----
        The ``store`` keyword is a dictionary, where each key specifies the name of the resulting data entry in the
        samples object, and the value is a callable which receives the ``context``, and returns a value from it.

        This means that the context can be inspected and arbitrarily summarised before storage. In particular, this
        allows for taking slices of arrays and saving them. One thing to note is that the context is dictionary-like,
        but is not a dictionary. The elements of the context are only available by using the ``get`` method, rather than
        directly subscripting the object like a normal dictionary.

        .. warning:: only scalars and arrays are supported for storage in the chain itself.
        """

        super().__init__(io_options.get("store", None))

        if ctx_variables is None:
            ctx_variables = ["brightness_temperature", "ionization_box"]

        self.redshift = redshift
        if not hasattr(self.redshift, "__len__"):
            self.redshift = [self.redshift]

        self.user_params = p21.UserParams(user_params)
        self.flag_options = p21.FlagOptions(flag_options)
        self.astro_params = p21.AstroParams(astro_params)
        self.cosmo_params = p21.CosmoParams(cosmo_params)
        self.change_seed_every_iter = change_seed_every_iter
        self.initial_conditions_seed = initial_conditions_seed

        self.regenerate = regenerate
        self.ctx_variables = ctx_variables

        self.z_step_factor = z_step_factor
        self.z_heat_max = z_heat_max

        self.io_options = {
            "store": {},  # (derived) quantities to store in the MCMC chain.
            "cache_dir": None,  # where full data sets will be written/read from.
            "cache_mcmc": False,  # whether to cache ionization data sets
            # (done before parameter retention step)
        }

        self.io_options.update(io_options)

        if self.initial_conditions_seed and self.change_seed_every_iter:
            logger.warning(
                "Attempting to set initial conditions seed while desiring to change seeds every iteration. Unsetting initial conditions seed."
            )
            self.initial_conditions_seed = None

    def setup(self):
        """
        Perform setup of the core.

        Notes
        -----
        This method is called automatically by its parent :class:`~LikelihoodComputationChain`, and should not be
        invoked directly.
        """
        super().setup()

        # If the chain has different parameter truths, we want to use those for our defaults.
        self.astro_params, self.cosmo_params = self._update_params(
            self.chain.createChainContext().getParams()
        )

        if self.z_heat_max is not None:
            p21.global_params.Z_HEAT_MAX = self.z_heat_max

        # Here we initialize the init and perturb boxes.
        # If modifying cosmo, we don't want to do this, because we'll create them
        # on the fly on every iteration.
        if (
            not any(p in self.cosmo_params.self.keys() for p in self.parameter_names)
            and not self.change_seed_every_iter
        ):
            logger.info("Initializing init and perturb boxes for the entire chain.")

            initial_conditions = p21.initial_conditions(
                user_params=self.user_params,
                cosmo_params=self.cosmo_params,
                direc=self.io_options["cache_dir"],
                regenerate=self.regenerate,
                random_seed=self.initial_conditions_seed,
            )

            # update the seed
            self.initial_conditions_seed = initial_conditions.random_seed

            perturb_field = []
            for z in self.redshift:
                perturb_field += [
                    p21.perturb_field(
                        redshift=z,
                        init_boxes=initial_conditions,
                        direc=self.io_options["cache_dir"],
                        regenerate=self.regenerate,
                    )
                ]
            logger.info("Initialization done.")

    def get_current_init_and_perturb(self, cosmo_params):
        initial_conditions = p21.initial_conditions(
            user_params=self.user_params,
            cosmo_params=cosmo_params,
            direc=self.io_options["cache_dir"],
            regenerate=False,
            random_seed=self.initial_conditions_seed,
            write=self.io_options["cache_mcmc"],
        )

        perturb_field = []
        for z in self.redshift:
            perturb_field += [
                p21.perturb_field(
                    redshift=z,
                    init_boxes=initial_conditions,
                    direc=self.io_options["cache_dir"],
                    regenerate=False,
                    write=self.io_options["cache_mcmc"],
                )
            ]

        return initial_conditions, perturb_field

    def build_model_data(self, ctx):
        # Update parameters
        logger.debug("Updating parameters: {}".format(ctx.getParams()))
        astro_params, cosmo_params = self._update_params(ctx.getParams())
        logger.debug("AstroParams: {}".format(astro_params))

        # Explicitly get the init and perturb boxes, because we don't want to
        # regenerate them (they will be read in *unless* we are changing
        # cosmo or seed)
        initial_conditions, perturb_field = self.get_current_init_and_perturb(
            cosmo_params
        )

        # Call C-code
        coeval = p21.run_coeval(
            redshift=self.redshift,
            astro_params=astro_params,
            flag_options=self.flag_options,
            init_box=initial_conditions,
            perturb=perturb_field,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate,
            random_seed=self.initial_conditions_seed,
            write=self.io_options["cache_mcmc"],
            direc=self.io_options["cache_dir"],
        )

        logger.debug("Adding {} to context data".format(self.ctx_variables))
        for key in self.ctx_variables:
            try:
                ctx.add(key, [getattr(c, key) for c in coeval])
            except KeyError:
                raise KeyError(
                    "ctx_variables must be drawn from the list ['init', 'perturb', "
                    "'ionization_box', 'brightness_temperature']"
                )

    def _update_params(self, params):
        """
        Update all the parameter structures which get passed to the driver, for this iteration.

        Parameters
        ----------
        params : Parameter object from cosmoHammer

        """
        ap_dict = copy.copy(self.astro_params.self)
        cp_dict = copy.copy(self.cosmo_params.self)

        ap_dict.update(
            **{
                k: getattr(params, k)
                for k, v in params.items()
                if k in self.astro_params.defining_dict
            }
        )
        cp_dict.update(
            **{
                k: getattr(params, k)
                for k, v in params.items()
                if k in self.cosmo_params.defining_dict
            }
        )

        return p21.AstroParams(**ap_dict), p21.CosmoParams(**cp_dict)


class CoreLightConeModule(CoreCoevalModule):
    """
    Core module for evaluating lightcone simulations.

    See :class:`~CoreCoevalModule` for info on all parameters, which are identical to this class, with the exception
    of `redshift`, which in this case must be a scalar.

    This module will add the following quantities to the context:

    1. ``lightcone``: a :class:`~py21cmmc._21cmfast.wrapper.LightCone` instance.
    """

    def __init__(self, *, max_redshift=None, **kwargs):
        if "ctx_variables" in kwargs:
            warnings.warn(
                "ctx_variables does not apply to the lightcone module (at least not yet). It will be ignored."
            )

        super().__init__(**kwargs)
        self.max_redshift = max_redshift

    @property
    def lightcone_slice_redshifts(self):
        """
        The redshift at each slice of the lightcone.
        """
        # noinspection PyProtectedMember
        return p21.wrapper._get_lightcone_redshifts(
            self.cosmo_params,
            self.max_redshift,
            self.redshift[0],
            self.user_params,
            self.z_step_factor,
        )

    def build_model_data(self, ctx):
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        # Call C-code
        lightcone = p21.run_lightcone(
            redshift=self.redshift[0],
            max_redshift=self.max_redshift,
            astro_params=astro_params,
            flag_options=self.flag_options,
            cosmo_params=cosmo_params,
            user_params=self.user_params,
            z_step_factor=self.z_step_factor,
            regenerate=self.regenerate,
            random_seed=self.initial_conditions_seed,
            write=self.io_options["cache_mcmc"],
            direc=self.io_options["cache_dir"],
        )

        ctx.add("lightcone", lightcone)


class CoreLuminosityFunction(CoreCoevalModule):
    """
    A Core Module that produces model luminosity functions at a range of redshifts.
    """

    def __init__(self, sigma, **kwargs):
        self._sigma = sigma
        super().__init__(**kwargs)

    def setup(self):
        CoreBase.setup(self)

        # If the chain has different parameter truths, we want to use those for our defaults.
        self.astro_params, self.cosmo_params = self._update_params(
            self.chain.createChainContext().getParams()
        )

    def run(self, astro_params, cosmo_params):
        return p21.compute_luminosity_function(
            redshifts=self.redshift,
            astro_params=astro_params,
            flag_options=self.flag_options,
            cosmo_params=cosmo_params,
            user_params=self.user_params,
        )

    def build_model_data(self, ctx):
        # Update parameters
        astro_params, cosmo_params = self._update_params(ctx.getParams())

        # Call C-code
        Muv, mhalo, lfunc = self.run(astro_params, cosmo_params)

        Muv = [m[~np.isnan(l)] for l, m in zip(lfunc, Muv)]
        mhalo = [m[~np.isnan(l)] for l, m in zip(lfunc, mhalo)]
        lfunc = [m[~np.isnan(l)] for l, m in zip(lfunc, lfunc)]

        ctx.add("luminosity_function", {"Muv": Muv, "mhalo": mhalo, "lfunc": lfunc})

    @property
    def sigma(self):
        if not hasattr(self._sigma, "__len__") or len(self._sigma) != len(
            self.redshift
        ):
            return [self._sigma] * len(self.redshift)
        else:
            return self._sigma

    def convert_model_to_mock(self, ctx):
        lfunc = ctx.get("luminosity_function")["lfunc"]
        muv = ctx.get("luminosity_function")["Muv"]

        for i, s in enumerate(self.sigma):  # each redshift
            try:
                lfunc[i] += np.random.normal(loc=0, scale=s(muv), size=len(lfunc[i]))
            except TypeError:
                lfunc[i] += np.random.normal(loc=0, scale=s, size=len(lfunc[i]))
