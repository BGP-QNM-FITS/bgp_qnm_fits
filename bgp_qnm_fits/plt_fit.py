import numpy as np
import qnmfits
import os 
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import time

from bgp_qnm_fits.base_fit import Base_BGP_fit
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

class BGP_PLT_fit(Base_BGP_fit):
    def __init__(
        self,
        *args,
        t0,
        use_nonlinear_params=False,
        decay_corrected=False, 
        num_samples=10000,
        quantiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.key = jax.random.PRNGKey(int(time.time()))
        self.num_samples = num_samples
        self.quantiles = quantiles
        self.use_nonlinear_params = use_nonlinear_params
        self.decay_corrected = decay_corrected

        if isinstance(t0, (float, int)):
            self.fit = self.get_fit_at_t0(t0)
        elif isinstance(t0, (list, tuple, np.ndarray)):
            self.fits = []
            for t0_val in tqdm(t0, desc="Fitting at t0 values"):
                self.fits.append(self.get_fit_at_t0(t0_val))
        else:
            raise ValueError("t0 must be a float, int, list or tuple")

    