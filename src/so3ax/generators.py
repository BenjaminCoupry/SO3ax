import jax

def generators_tensor(l):
    """Build a basis of the `so(3)` Lie algebra in the `l` irreducible representation.

    The three generators are `(-i*Jx, -i*Jy, -i*Jz)`, the anti-Hermitian angular
    momentum operators in the standard `(x, y, z)` frame, and satisfy
    `[G[a], G[b]] = eps_{abc} G[c]`.

    Parameters
    ----------
    l : int
        Angular momentum quantum number of the representation (non-negative
        integer).

    Returns
    -------
    jax.numpy.ndarray
        Complex tensor of shape `(3, 2*l+1, 2*l+1)` stacking the generators of
        rotation about `x`, `y` and `z` along the first axis.
    """
    m_r = jax.numpy.arange(-l, l)
    m_l = jax.numpy.arange(-l + 1, l + 1)
    m_z = jax.numpy.arange(-l, l + 1)
    raising = jax.numpy.diag(jax.numpy.sqrt(l * (l + 1) - m_r * (m_r + 1)), k=-1)
    lowering = jax.numpy.diag(jax.numpy.sqrt(l * (l + 1) - m_l * (m_l - 1)), k=1)
    G_x = -1j * 0.5 * (raising + lowering)
    G_y = 0.5 * (lowering - raising)
    G_z = -1j * jax.numpy.diag(m_z)
    generators = jax.numpy.stack([G_x, G_y, G_z], axis=0)
    return generators