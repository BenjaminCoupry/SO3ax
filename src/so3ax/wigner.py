import jax
import so3ax.generators as generators

def wigner_D_tensor(l, log_coordinates):
    """Compute Wigner D-matrices of the `l` irreducible representation from rotations
    given in logarithmic coordinates.

    Parameters
    ----------
    l : int
        Angular momentum quantum number of the representation (non-negative
        integer).
    log_coordinates : jax.numpy.ndarray
        Real array of shape `(..., 3)`, the coordinates of each rotation on the
        `so(3)` Lie algebra basis returned by `generators_tensor` (its norm is
        the rotation angle).

    Returns
    -------
    jax.numpy.ndarray
        Complex tensor of shape `(..., 2*l+1, 2*l+1)`, the Wigner D-matrix of
        each rotation.
    """
    generators_tensor = generators.generators_tensor(l)
    product = jax.numpy.einsum("...a, aij-> ...ij", log_coordinates, generators_tensor)
    wigner_D = jax.scipy.linalg.expm(product)
    return wigner_D
