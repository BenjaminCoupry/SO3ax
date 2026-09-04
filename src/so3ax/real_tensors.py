import functools
import jax
import so3ax.clebsch_gordan as clebsch_gordan
import so3ax.generators as generators
import so3ax.wigner as wigner

def real_to_complex(l):
    """Build the unitary change-of-basis matrix from real to complex spherical
    harmonics for the `l` irreducible representation.

    The real basis is the standard real spherical harmonics ordered by
    `m = -l, ..., l` (`(y, z, x)` for `l = 1`)

    Parameters
    ----------
    l : int
        Angular momentum quantum number of the representation (non-negative
        integer).

    Returns
    -------
    jax.numpy.ndarray
        Complex matrix `q` of shape `(2*l+1, 2*l+1)` mapping real-basis
        coordinates to complex-basis coordinates (`v_complex = q @ v_real`),
        including the `(-i)**l` phase that keeps the derived real tensors real.
    """
    m = jax.numpy.arange(1, l + 1)
    s = jax.numpy.reciprocal(jax.numpy.sqrt(2.0))
    sign = jax.numpy.pow(-1.0, m)
    q = jax.numpy.zeros((2 * l + 1, 2 * l + 1), dtype=jax.numpy.complex128)
    q = q.at[l, l].set(1.0)
    q = q.at[l - m, l + m].set(s)
    q = q.at[l + m, l + m].set(sign * s)
    q = q.at[l - m, l - m].set(1j * s)
    q = q.at[l + m, l - m].set(-1j * sign * s)
    real_q = jax.numpy.pow(-1j, l) * q
    return real_q

@functools.partial(jax.jit, static_argnames=('l1','l2','l3'))
def clebsch_gordan_real_tensor(l1, l2, l3):
    """Convert the Clebsch-Gordan tensor `l1 (x) l2 -> l3` to the real spherical
    harmonic basis.

    Parameters
    ----------
    l1, l2, l3 : int
        Angular momentum quantum numbers of the two coupled representations and
        of the resulting representation.

    Returns
    -------
    jax.numpy.ndarray
        Real tensor of shape `(2*l1+1, 2*l2+1, 2*l3+1)`, the real-basis
        counterpart of `clebsch_gordan_tensor`.
    """
    clebsch_gordan_tensor = clebsch_gordan.clebsch_gordan_tensor(l1, l2, l3)
    q1, q2, q3 = real_to_complex(l1), real_to_complex(l2), real_to_complex(l3)
    result = jax.numpy.real(jax.numpy.einsum("ij,kl,mn,ikn->jlm", q1, q2, jax.numpy.conj(q3.T), clebsch_gordan_tensor))
    return result

@functools.partial(jax.jit, static_argnames=('l'))
def generators_real_tensor(l):
    """Convert the `so(3)` generators of the `l` irreducible representation to the
    real spherical harmonic basis.

    Parameters
    ----------
    l : int
        Angular momentum quantum number of the representation (non-negative
        integer).

    Returns
    -------
    jax.numpy.ndarray
        Real tensor of shape `(3, 2*l+1, 2*l+1)`, the real-basis counterpart of
        `generators_tensor` (each slice is a real antisymmetric matrix).
    """
    generators_tensor = generators.generators_tensor(l)
    q = real_to_complex(l)
    result = jax.numpy.real(jax.numpy.einsum('ij, lm, kjl -> kim', jax.numpy.conj(q.T), q, generators_tensor))
    return result

@functools.partial(jax.jit, static_argnames=('l'))
def wigner_D_real_tensor(l, log_coordinates):
    """Compute real Wigner D-matrices of the `l` irreducible representation from
    rotations given in logarithmic coordinates.

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
        Real tensor of shape `(..., 2*l+1, 2*l+1)`, the real-basis Wigner
        D-matrix of each rotation.
    """
    wigner_D_tensor = wigner.wigner_D_tensor(l, log_coordinates)
    q = real_to_complex(l)
    result = jax.numpy.real(jax.numpy.einsum('ij, lm, ...jl -> ...im', jax.numpy.conj(q.T), q, wigner_D_tensor))
    return result

def so3_to_log_coordinates(rotation):
    """Recover the logarithmic coordinates of a rotation from its `SO(3)` matrix.

    Parameters
    ----------
    rotation : jax.numpy.ndarray
        Real array of shape `(..., 3, 3)` of rotation matrices in the standard
        `(x, y, z)` frame.

    Returns
    -------
    jax.numpy.ndarray
        Real array of shape `(..., 3)`, the logarithmic coordinates of each
        rotation.
    """
    axis = jax.numpy.stack([rotation[..., 2, 1] - rotation[..., 1, 2],
                            rotation[..., 0, 2] - rotation[..., 2, 0],
                            rotation[..., 1, 0] - rotation[..., 0, 1]], axis=-1)
    axis_norm = jax.numpy.linalg.norm(axis, axis=-1, keepdims=True)
    trace = jax.numpy.trace(rotation, axis1=-2, axis2=-1)[..., None]
    angle = jax.numpy.arctan2(0.5 * axis_norm, 0.5 * (trace - 1.0))
    log_coordinates = angle * axis / jax.numpy.where(axis_norm > 0.0, axis_norm, 1.0) # pyright: ignore[reportOperatorIssue]
    return log_coordinates
