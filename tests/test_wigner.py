import jax
import pytest

from so3ax.generators import generators_tensor
from so3ax.real_tensors import so3_to_log_coordinates
from so3ax.wigner import wigner_D_tensor


ATOL = 1e-6 if jax.numpy.result_type(float) == jax.numpy.float64 else 1e-3
N_TEST = 15
LMAX = 8 if jax.numpy.result_type(float) == jax.numpy.float64 else 5

wigner_D_tensor = jax.jit(wigner_D_tensor, static_argnums=(0,))


def _random_l(key):
    return int(jax.random.randint(key, (), 0, LMAX + 1))


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_has_expected_shape(seed):
    l = _random_l(jax.random.key(seed))

    single = wigner_D_tensor(l, jax.numpy.zeros(3))

    assert single.shape == (2 * l + 1, 2 * l + 1)
    assert jax.numpy.iscomplexobj(single)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_is_identity_at_zero(seed):
    l = _random_l(jax.random.key(seed))

    identity = wigner_D_tensor(l, jax.numpy.zeros(3))

    assert jax.numpy.allclose(identity, jax.numpy.eye(2 * l + 1), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_is_special_unitary(seed):
    key_l, key_vector = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    coordinates = jax.random.normal(key_vector, (3,))

    wigner_D = wigner_D_tensor(l, coordinates)

    assert jax.numpy.allclose(
        wigner_D @ jax.numpy.conj(wigner_D.T), jax.numpy.eye(2 * l + 1), atol=ATOL
    )
    assert jax.numpy.allclose(jax.numpy.linalg.det(wigner_D), 1.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_inverse_is_the_negated_coordinates(seed):
    key_l, key_vector = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    coordinates = jax.random.normal(key_vector, (3,))

    forward = wigner_D_tensor(l, coordinates)
    backward = wigner_D_tensor(l, -coordinates)

    assert jax.numpy.allclose(forward @ backward, jax.numpy.eye(2 * l + 1), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_is_a_group_homomorphism(seed):
    key_l, key_first, key_second = jax.random.split(jax.random.key(seed), 3)
    l = _random_l(key_l)
    orthogonal_first = jax.random.orthogonal(key_first, 3)
    orthogonal_second = jax.random.orthogonal(key_second, 3)
    first = orthogonal_first.at[:, 0].multiply(jax.numpy.linalg.det(orthogonal_first))
    second = orthogonal_second.at[:, 0].multiply(jax.numpy.linalg.det(orthogonal_second))

    product = wigner_D_tensor(l, so3_to_log_coordinates(second)) @ wigner_D_tensor(
        l, so3_to_log_coordinates(first)
    )
    composed = wigner_D_tensor(l, so3_to_log_coordinates(second @ first))

    assert jax.numpy.allclose(product, composed, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_matches_the_generators_at_first_order(seed):
    l = _random_l(jax.random.key(seed))

    jacobian = jax.jacfwd(lambda coordinates: wigner_D_tensor(l, coordinates))(jax.numpy.zeros(3))

    assert jax.numpy.allclose(
        jax.numpy.moveaxis(jacobian, -1, 0), generators_tensor(l), atol=ATOL
    )


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_is_diagonal_for_a_z_rotation(seed):
    key_l, key_angle = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    angle = jax.random.uniform(key_angle, (), minval=-jax.numpy.pi, maxval=jax.numpy.pi)

    wigner_D = wigner_D_tensor(l, angle * jax.numpy.array([0.0, 0.0, 1.0]))
    weights = jax.numpy.arange(-l, l + 1)
    expected = jax.numpy.diag(jax.numpy.exp(-1j * angle * weights))

    assert jax.numpy.allclose(wigner_D, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_wigner_D_tensor_is_two_pi_periodic(seed):
    key_l, key_direction = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)
    direction = jax.random.normal(key_direction, (3,))
    axis = direction / jax.numpy.linalg.norm(direction)

    full_turn = wigner_D_tensor(l, 2.0 * jax.numpy.pi * axis)

    assert jax.numpy.allclose(full_turn, jax.numpy.eye(2 * l + 1), atol=ATOL)
