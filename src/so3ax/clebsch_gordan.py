import jax


def _factorial_quotient(numerator, denominator, power):
    """Compute `(prod_i numerator[i]! / prod_j denominator[j]!) ** power`,
    the quotient of factorial products, raised to a power.

    Parameters
    ----------
    numerator : jax.numpy.ndarray
        Integer array whose factorials appear in the numerator.
    denominator : jax.numpy.ndarray
        Integer array whose factorials appear in the denominator.
    power : float
        Exponent applied to the overall quotient.

    Returns
    -------
    jax.numpy.ndarray
        The quotient (scalar), raised to `power`.
    """
    kmax = jax.numpy.maximum(jax.numpy.max(numerator), jax.numpy.max(denominator))
    f = lambda k, s : s + (jax.numpy.sum(numerator >= k) - jax.numpy.sum(denominator >= k))*jax.numpy.log(jax.numpy.float64(k))
    log_quotient = jax.lax.fori_loop(1, kmax+1, f, 0, unroll=False)
    quotient = jax.numpy.exp(power*log_quotient)
    return quotient

def _clebsch_gordan_coefficient(l1, m1, l2, m2, l3, m3):
    """Compute a Clebsch-Gordan coefficient `<l1 m1, l2 m2 | l3 m3>`.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers of the two coupled representations
        and of the resulting representation.
    m1, m2, m3 : int
        Associated magnetic quantum numbers, with `|mi| <= li`.

    Returns
    -------
    jax.numpy.ndarray
        The Clebsch-Gordan coefficient (scalar), zero if the selection rules
        `(|m1+m2| <= l3, |l1-l2| <= l3 <= l1+l2, m3 = m1+m2)` are not satisfied.
    """
    c = _factorial_quotient(jax.numpy.asarray([l3 + l1 - l2, l3 - l1 + l2, l1 + l2 - l3, l3 + m3, l3 - m3]), 
                           jax.numpy.asarray([l1 + l2 + l3 + 1, l1 - m1, l1 + m1, l2 - m2, l2 + m2]), 0.5)
    f = lambda v, s : s + _factorial_quotient(jax.numpy.asarray([l2 + l3 + m1 - v, l1 - m1 + v]), 
                        jax.numpy.asarray([v, l3 - l1 + l2 - v, l3 + m3 - v, v + l1 - l2 - m3]), 1.0) * jax.numpy.pow(-1, v + l2 + m2)
    vmin = jax.numpy.maximum(jax.numpy.maximum(-l1 + l2 + m3, -l1 + m1) , 0)
    vmax = jax.numpy.minimum(jax.numpy.minimum(l2 + l3 + m1, l3 - l1 + l2) , l3 + m3)
    s = jax.lax.fori_loop(vmin, vmax+1, f, 0, unroll=False)
    condition_a = jax.numpy.abs(m1 + m2) <= l3
    condition_b = jax.numpy.logical_and(l3 >= jax.numpy.abs(l1 - l2), l3 <= jax.numpy.abs(l1 + l2))
    condition_c = m3 == m1+m2
    condition = jax.numpy.all(jax.numpy.stack([condition_a, condition_b, condition_c],axis=-1),axis=-1)
    cg = jax.numpy.where(condition, c * s * jax.numpy.sqrt(2 * l3 + 1), 0)
    return cg

def clebsch_gordan_tensor(l1, l2, l3):
    """Build the full Clebsch-Gordan tensor coupling `l1 (x) l2 -> l3`.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers of the two coupled representations
        and of the resulting representation.

    Returns
    -------
    jax.numpy.ndarray
        Tensor of shape `(2*l1+1, 2*l2+1, 2*l3+1)` whose entry
        `[l1+m1, l2+m2, l3+m3]` holds `<l1 m1, l2 m2 | l3 m3>`
        (zero outside of `m3 = m1 + m2`).
    """
    m1 = jax.numpy.repeat(jax.numpy.arange(-l1, l1 + 1), 2*l2+1)
    m2 = jax.numpy.tile(jax.numpy.arange(-l2, l2 + 1), 2*l1+1)
    m3 = m1+m2
    coefficients = jax.vmap(_clebsch_gordan_coefficient, in_axes=(None, 0, None, 0, None, 0))(l1, m1, l2, m2, l3, m3)
    clebsch_gordan = jax.numpy.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)).at[l1+m1, l2+m2, l3+m3].set(coefficients)
    return clebsch_gordan