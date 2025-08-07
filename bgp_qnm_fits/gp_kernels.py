import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def squared_exp_element(t1, t2, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-0.5 * dist**2 / period**2)

def kernel_Wendland(x1, x2, sigmasq=1.0, l=1.0, D=1, q=0):
  l, D, q = float(l), int(D), int(q)
  K = jnp.zeros((len(x1), len(x2)))
  j = int(D/2) + q + 1
  tau = jnp.abs(x1[:,jnp.newaxis]-x2[jnp.newaxis,:]) / jnp.abs(l)
  if q==0:
    K += jnp.maximum(0, 1-tau)**(j)
  elif q==1:
    K += jnp.maximum(0, 1-tau)**(j+1) * ( (j+1)*tau + 1 )
  elif q==2:
    K += jnp.maximum(0, 1-tau)**(j+2) * ( (j**2+4*j+3)*tau**2 +
                                (3*j+6)*tau + 3 ) / 3
  elif q==3:
    K += jnp.maximum(0, 1-tau)**(j+3) * ( (j**3+9*j**2+23*j+15)*tau**3 +
                            (6*j**2+36*j+45)*tau**2 + (15*j+45)*tau  + 15 ) / 15
  else:
    raise ValueError('Not implemented q={}'.format(q))
  return sigmasq * K

def exponential_func(t, length_scale, t_s, sigma_max):
    return sigma_max * jnp.exp(-(t - t_s) / length_scale)

def smoothmax(x, x_max, smoothness):
    return (x + x_max - jnp.sqrt((x - x_max) ** 2 + smoothness * x_max**2)) * 0.5


def new_func(t, length_scale, t_s, sigma_max, A_max, smoothness):
    t = jnp.asarray(t)
    return jnp.exp(
        smoothmax(
            jnp.log(exponential_func(t, length_scale, t_s, sigma_max)),
            jnp.log(1.1 * A_max), # TODO ensure conservative estimate 
            smoothness,
        )
    )


def periodic_kernel(t1, t2, length_scale, period):
    dist = jnp.abs(t1[:, None] - t2[None, :])
    return jnp.exp(-2 * jnp.sin(jnp.pi * dist / period) ** 2 / length_scale**2)


def kernel_test(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return kwargs["sigma_max"] ** 2 * squared_exp_element(t1, t2, kwargs["period"])


def kernel_WN(analysis_times, **kwargs):
    return kwargs["sigma_max"] ** 2 * jnp.eye(len(analysis_times))


def kernel_GP(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        #squared_exp_element(t1, t2, kwargs["period"])
        kernel_Wendland(t1, t2, sigmasq=1.0, l=kwargs["period"], q=3)
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["A_max"],
            kwargs["smoothness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["A_max"],
            kwargs["smoothness"],
        )[None, :]
    )


def kernel_GPC(analysis_times, **kwargs):
    t1 = analysis_times
    t2 = analysis_times
    return (
        (
            #squared_exp_element(t1, t2, kwargs["period"]) ** kwargs["a"]
            kernel_Wendland(t1, t2, sigmasq=1.0, l=kwargs["period"], q=3) ** kwargs["a"]
            * periodic_kernel(t1, t2, kwargs["length_scale_2"], kwargs["period_2"]) ** (1 - kwargs["a"])
        )
        * new_func(
            t1,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["A_max"],
            kwargs["smoothness"],
        )[:, None]
        * new_func(
            t2,
            kwargs["length_scale"],
            kwargs["t_s"],
            kwargs["sigma_max"],
            kwargs["A_max"],
            kwargs["smoothness"],
        )[None, :]
    )


def compute_kernel_matrix(analysis_times, hyperparams, kernel, regularization_factor=1e2):
    return (
        kernel(jnp.asarray(analysis_times), **hyperparams)
        + jnp.eye(len(analysis_times)) * (hyperparams["A_min_reg"] * regularization_factor) ** 2
    )
