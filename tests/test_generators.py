import jax
import pytest

from so3ax.generators import generators_tensor


ATOL = 1e-6
N_TEST = 15
LMAX = 8


def _random_l(key):
    return int(jax.random.randint(key, (), 0, LMAX + 1))


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_has_expected_shape(seed):
    l = _random_l(jax.random.key(seed))

    generators = generators_tensor(l)

    assert generators.shape == (3, 2 * l + 1, 2 * l + 1)
    assert jax.numpy.iscomplexobj(generators)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_is_traceless(seed):
    l = _random_l(jax.random.key(seed))

    traces = jax.numpy.trace(generators_tensor(l), axis1=-2, axis2=-1)

    assert jax.numpy.allclose(traces, 0.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_is_anti_hermitian(seed):
    l = _random_l(jax.random.key(seed))

    generators = generators_tensor(l)
    hermitian_conjugate = jax.numpy.conj(jax.numpy.swapaxes(generators, -1, -2))

    assert jax.numpy.allclose(hermitian_conjugate, -generators, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_cartan_generator_carries_the_weights(seed):
    l = _random_l(jax.random.key(seed))

    cartan = generators_tensor(l)[2]
    expected = jax.numpy.diag(-1j * jax.numpy.arange(-l, l + 1))

    assert jax.numpy.allclose(cartan, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_components_share_the_integer_spectrum(seed):
    l = _random_l(jax.random.key(seed))

    generators = generators_tensor(l)
    expected = jax.numpy.arange(-l, l + 1).astype(generators.real.dtype)

    for component in generators:
        spectrum = jax.numpy.linalg.eigvalsh(1j * component)
        assert jax.numpy.allclose(spectrum, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_satisfies_so3_commutation_relations(seed):
    l = _random_l(jax.random.key(seed))

    generator_x, generator_y, generator_z = generators_tensor(l)
    commutator = lambda a, b: a @ b - b @ a

    assert jax.numpy.allclose(commutator(generator_x, generator_y), generator_z, atol=ATOL)
    assert jax.numpy.allclose(commutator(generator_y, generator_z), generator_x, atol=ATOL)
    assert jax.numpy.allclose(commutator(generator_z, generator_x), generator_y, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_casimir_is_a_scalar_matrix(seed):
    l = _random_l(jax.random.key(seed))

    generators = generators_tensor(l)
    casimir = jax.numpy.einsum("aij, ajk -> ik", generators, generators)

    assert jax.numpy.allclose(casimir, -l * (l + 1) * jax.numpy.eye(2 * l + 1), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_generators_tensor_exponentiates_to_special_unitary(seed):
    key_l, key_vector = jax.random.split(jax.random.key(seed), 2)
    l = _random_l(key_l)

    rotation_vector = jax.random.normal(key_vector, (3,))
    algebra_element = jax.numpy.einsum("a, aij -> ij", rotation_vector, generators_tensor(l))
    group_element = jax.scipy.linalg.expm(algebra_element)

    assert jax.numpy.allclose(
        group_element @ jax.numpy.conj(group_element.T),
        jax.numpy.eye(2 * l + 1),
        atol=ATOL,
    )
    assert jax.numpy.allclose(jax.numpy.linalg.det(group_element), 1.0, atol=ATOL)


def test_generators_tensor_matches_the_spin_one_reference():
    generators = generators_tensor(1)

    s = 1.0 / jax.numpy.sqrt(2.0)
    generator_x = -1j * s * jax.numpy.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    generator_y = s * jax.numpy.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    generator_z = jax.numpy.array([[1j, 0, 0], [0, 0, 0], [0, 0, -1j]])
    reference = jax.numpy.stack([generator_x, generator_y, generator_z])

    assert jax.numpy.allclose(generators, reference, atol=ATOL)
