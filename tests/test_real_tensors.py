import jax
import pytest
import sphericart.jax

from so3ax.real_tensors import (
    clebsch_gordan_real_tensor,
    generators_real_tensor,
    real_to_complex,
    so3_to_log_coordinates,
    wigner_D_real_tensor,
)
from so3ax.wigner import wigner_D_tensor


ATOL = 1e-6
N_TEST = 15
LMAX = 6

# jit-cached across the seed parametrization; jit is transparent to the results.
real_to_complex = jax.jit(real_to_complex, static_argnums=(0,))


def _classic_axis_rotation(axis, angle):
    """Classic 3x3 rotation matrix about `x` (0), `y` (1) or `z` (2)."""
    cosine, sine = jax.numpy.cos(angle), jax.numpy.sin(angle)
    return [
        jax.numpy.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]),
        jax.numpy.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]),
        jax.numpy.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]),
    ][axis]


def _rotation_matrix(rotation_vector):
    """Classic 3x3 rotation matrix (Rodrigues) for a rotation vector."""
    angle = jax.numpy.linalg.norm(rotation_vector)
    x, y, z = rotation_vector / angle
    skew = jax.numpy.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        jax.numpy.eye(3)
        + jax.numpy.sin(angle) * skew
        + (1.0 - jax.numpy.cos(angle)) * (skew @ skew)
    )


def _random_l(key):
    return int(jax.random.randint(key, (), 0, LMAX + 1))


def _random_triple(key):
    key_l1, key_l2, key_l3 = jax.random.split(key, 3)
    l1, l2 = _random_l(key_l1), _random_l(key_l2)
    l3 = int(jax.random.randint(key_l3, (), abs(l1 - l2), l1 + l2 + 1))
    return l1, l2, l3


def _unit_points(key, count):
    points = jax.random.normal(key, (count, 3))
    return points / jax.numpy.linalg.norm(points, axis=-1, keepdims=True)


def _spherical_harmonics_block(points, l):
    harmonics = sphericart.jax.spherical_harmonics(points, LMAX)
    return harmonics[..., l * l : l * l + 2 * l + 1]


@pytest.mark.parametrize("seed", range(N_TEST))
def test_real_to_complex_is_unitary(seed):
    l = _random_l(jax.random.key(seed))

    q = real_to_complex(l)

    assert q.shape == (2 * l + 1, 2 * l + 1)
    assert jax.numpy.allclose(q @ jax.numpy.conj(q.T), jax.numpy.eye(2 * l + 1), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_real_tensor_is_real_and_antisymmetric(seed):
    l = _random_l(jax.random.key(seed))

    generators = generators_real_tensor(l)

    assert generators.shape == (3, 2 * l + 1, 2 * l + 1)
    assert not jax.numpy.iscomplexobj(generators)
    assert jax.numpy.allclose(generators + jax.numpy.swapaxes(generators, -1, -2), 0.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_real_tensor_satisfies_so3_commutation_relations(seed):
    l = _random_l(jax.random.key(seed))

    generator_x, generator_y, generator_z = generators_real_tensor(l)
    commutator = lambda a, b: a @ b - b @ a

    assert jax.numpy.allclose(commutator(generator_x, generator_y), generator_z, atol=ATOL)
    assert jax.numpy.allclose(commutator(generator_y, generator_z), generator_x, atol=ATOL)
    assert jax.numpy.allclose(commutator(generator_z, generator_x), generator_y, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_real_tensor_is_special_orthogonal(seed):
    key_l, key_rotation = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    rotation_vector = jax.random.normal(key_rotation, (3,))

    wigner_D_real = wigner_D_real_tensor(l, rotation_vector)

    assert jax.numpy.allclose(wigner_D_real @ wigner_D_real.T, jax.numpy.eye(2 * l + 1), atol=ATOL)
    assert jax.numpy.allclose(jax.numpy.linalg.det(wigner_D_real), 1.0, atol=ATOL)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_real_tensor_is_the_classic_axis_rotation_for_l1(axis, seed):
    angle = jax.random.uniform(
        jax.random.key(seed), (), minval=-jax.numpy.pi, maxval=jax.numpy.pi
    )
    reorder = jax.numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    wigner_D_real = wigner_D_real_tensor(1, angle * jax.numpy.eye(3)[axis])
    expected = reorder @ _classic_axis_rotation(axis, angle) @ reorder.T

    assert jax.numpy.allclose(wigner_D_real, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_real_tensor_is_an_intertwiner(seed):
    key_triple, key_rotation = jax.random.split(jax.random.key(seed), 2)
    l1, l2, l3 = _random_triple(key_triple)
    rotation_vector = jax.random.normal(key_rotation, (3,))

    clebsch_gordan = clebsch_gordan_real_tensor(l1, l2, l3)
    wigner_D_1 = wigner_D_real_tensor(l1, rotation_vector)
    wigner_D_2 = wigner_D_real_tensor(l2, rotation_vector)
    wigner_D_3 = wigner_D_real_tensor(l3, rotation_vector)

    coupled = jax.numpy.einsum("mM, nN, MNk -> mnk", wigner_D_1, wigner_D_2, clebsch_gordan)
    rotated = jax.numpy.einsum("mnK, Kk -> mnk", clebsch_gordan, wigner_D_3)

    assert jax.numpy.allclose(coupled, rotated, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_real_tensor_rotates_spherical_harmonics(seed):
    key_l, key_rotation, key_points = jax.random.split(jax.random.key(seed), 3)
    l = _random_l(key_l)
    rotation_vector = jax.random.normal(key_rotation, (3,))
    rotation = _rotation_matrix(rotation_vector)
    points = _unit_points(key_points, 32)

    rotated_harmonics = _spherical_harmonics_block(points @ rotation.T, l)
    expected = _spherical_harmonics_block(points, l) @ wigner_D_real_tensor(l, rotation_vector).T

    assert jax.numpy.allclose(rotated_harmonics, expected, atol=ATOL)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_real_tensor_generates_infinitesimal_rotations_of_spherical_harmonics(axis, seed):
    key_l, key_points = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    points = _unit_points(key_points, 32)

    def rotated_harmonics(angle):
        return _spherical_harmonics_block(points @ _classic_axis_rotation(axis, angle).T, l)

    derivative = jax.jacfwd(rotated_harmonics)(0.0)
    expected = _spherical_harmonics_block(points, l) @ generators_real_tensor(l)[axis].T

    assert jax.numpy.allclose(derivative, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_real_tensor_couples_spherical_harmonics_equivariantly(seed):
    key_triple, key_rotation, key_point = jax.random.split(jax.random.key(seed), 3)
    l1, l2, l3 = _random_triple(key_triple)
    rotation_vector = jax.random.normal(key_rotation, (3,))
    rotation = _rotation_matrix(rotation_vector)
    point = _unit_points(key_point, 1)[0]

    clebsch_gordan = clebsch_gordan_real_tensor(l1, l2, l3)
    coupled = jax.numpy.einsum(
        "a, b, abc -> c",
        _spherical_harmonics_block(point, l1),
        _spherical_harmonics_block(point, l2),
        clebsch_gordan,
    )
    coupled_rotated = jax.numpy.einsum(
        "a, b, abc -> c",
        _spherical_harmonics_block(rotation @ point, l1),
        _spherical_harmonics_block(rotation @ point, l2),
        clebsch_gordan,
    )

    assert jax.numpy.allclose(
        coupled_rotated, wigner_D_real_tensor(l3, rotation_vector) @ coupled, atol=ATOL
    )


def test_so3_to_log_coordinates_of_the_identity_is_zero():
    log_coordinates = so3_to_log_coordinates(jax.numpy.eye(3))

    assert jax.numpy.allclose(log_coordinates, jax.numpy.zeros(3), atol=ATOL)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("seed", range(N_TEST))
def test_so3_to_log_coordinates_recovers_a_classic_axis_rotation(axis, seed):
    angle = jax.random.uniform(jax.random.key(seed), (), minval=-3.0, maxval=3.0)

    log_coordinates = so3_to_log_coordinates(_classic_axis_rotation(axis, angle))

    assert jax.numpy.allclose(log_coordinates, angle * jax.numpy.eye(3)[axis], atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_so3_to_log_coordinates_inverts_the_l1_representation(seed):
    orthogonal = jax.random.orthogonal(jax.random.key(seed), 3)
    rotation = orthogonal.at[:, 0].multiply(jax.numpy.linalg.det(orthogonal))
    reorder = jax.numpy.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    reconstructed = wigner_D_real_tensor(1, so3_to_log_coordinates(rotation))
    expected = reorder @ rotation @ reorder.T

    assert jax.numpy.linalg.norm(so3_to_log_coordinates(rotation)) <= jax.numpy.pi + ATOL
    assert jax.numpy.allclose(reconstructed, expected, atol=ATOL)
