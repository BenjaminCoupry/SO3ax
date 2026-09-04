import jax
import pytest

from so3ax.clebsch_gordan import clebsch_gordan_tensor
from so3ax.wigner import wigner_D_tensor


ATOL = 1e-6
N_TEST = 15
LMAX = 5

clebsch_gordan_tensor = jax.jit(clebsch_gordan_tensor, static_argnums=(0, 1, 2))
wigner_D_tensor = jax.jit(wigner_D_tensor, static_argnums=(0,))


def _random_pair(key):
    key_l1, key_l2 = jax.random.split(key, 2)
    l1 = int(jax.random.randint(key_l1, (), 0, LMAX + 1))
    l2 = int(jax.random.randint(key_l2, (), 0, LMAX + 1))
    return l1, l2


def _random_triple(key):
    key_pair, key_l3 = jax.random.split(key, 2)
    l1, l2 = _random_pair(key_pair)
    l3 = int(jax.random.randint(key_l3, (), abs(l1 - l2), l1 + l2 + 1))
    return l1, l2, l3


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_has_expected_shape(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l3)

    assert clebsch_gordan.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_is_supported_on_m3_equals_m1_plus_m2(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l3)
    m1 = jax.numpy.arange(-l1, l1 + 1)[:, None, None]
    m2 = jax.numpy.arange(-l2, l2 + 1)[None, :, None]
    m3 = jax.numpy.arange(-l3, l3 + 1)[None, None, :]
    off_selection = jax.numpy.where(m3 == m1 + m2, 0.0, clebsch_gordan)

    assert jax.numpy.allclose(off_selection, 0.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_vanishes_outside_the_triangle(seed):
    key_l1, key_gap, key_below, key_above = jax.random.split(jax.random.key(seed), 4)
    l1 = int(jax.random.randint(key_l1, (), 0, LMAX + 1))
    gap = int(jax.random.randint(key_gap, (), 1, LMAX + 1))
    l2 = l1 + gap
    below = int(jax.random.randint(key_below, (), 0, gap))  # l3 < |l1 - l2|
    above = l1 + l2 + 1 + int(jax.random.randint(key_above, (), 0, LMAX + 1))

    assert jax.numpy.allclose(clebsch_gordan_tensor(l1, l2, below), 0.0, atol=ATOL)
    assert jax.numpy.allclose(clebsch_gordan_tensor(l1, l2, above), 0.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_stretched_state_is_one(seed):
    l1, l2 = _random_pair(jax.random.key(seed))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l1 + l2)

    assert jax.numpy.allclose(clebsch_gordan[-1, -1, -1], 1.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_scalar_coupling_special_case(seed):
    l = int(jax.random.randint(jax.random.key(seed), (), 0, LMAX + 1))

    scalar_slice = clebsch_gordan_tensor(l, l, 0)[:, :, 0]
    m = jax.numpy.arange(-l, l + 1)
    expected = jax.numpy.diag((-1.0) ** (l - m) / jax.numpy.sqrt(2 * l + 1))[:, ::-1]

    assert jax.numpy.allclose(scalar_slice, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_zero_projection_vanishes_for_odd_parity(seed):
    key_l1, key_l2, key_l3 = jax.random.split(jax.random.key(seed), 3)
    l1 = int(jax.random.randint(key_l1, (), 1, LMAX + 1))
    l2 = int(jax.random.randint(key_l2, (), 1, LMAX + 1))
    l3 = abs(l1 - l2) + 1 + 2 * int(jax.random.randint(key_l3, (), 0, min(l1, l2)))
    assert (l1 + l2 + l3) % 2 == 1

    zero_projection = clebsch_gordan_tensor(l1, l2, l3)[l1, l2, l3]

    assert jax.numpy.allclose(zero_projection, 0.0, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_rows_are_orthonormal(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l3)
    gram = jax.numpy.einsum("abk, abK -> kK", clebsch_gordan, clebsch_gordan)

    assert jax.numpy.allclose(gram, jax.numpy.eye(2 * l3 + 1), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_blocks_form_an_orthonormal_basis(seed):
    l1, l2 = _random_pair(jax.random.key(seed))
    dimension = (2 * l1 + 1) * (2 * l2 + 1)

    basis = jax.numpy.concatenate(
        [
            clebsch_gordan_tensor(l1, l2, l3).reshape(dimension, -1)
            for l3 in range(abs(l1 - l2), l1 + l2 + 1)
        ],
        axis=1,
    )

    assert jax.numpy.allclose(basis @ basis.T, jax.numpy.eye(dimension), atol=ATOL)
    assert jax.numpy.allclose(basis.T @ basis, jax.numpy.eye(dimension), atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_swap_symmetry(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    direct = clebsch_gordan_tensor(l1, l2, l3)
    swapped = jax.numpy.swapaxes(clebsch_gordan_tensor(l2, l1, l3), 0, 1)

    assert jax.numpy.allclose(direct, (-1.0) ** (l1 + l2 - l3) * swapped, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_sign_reversal_symmetry(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l3)
    reversed_m = clebsch_gordan[::-1, ::-1, ::-1]

    assert jax.numpy.allclose(clebsch_gordan, (-1.0) ** (l1 + l2 - l3) * reversed_m, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_column_exchange_symmetry(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    direct = clebsch_gordan_tensor(l1, l2, l3)
    exchanged = clebsch_gordan_tensor(l1, l3, l2)
    sign = ((-1.0) ** jax.numpy.arange(2 * l1 + 1))[:, None, None]
    factor = jax.numpy.sqrt((2 * l3 + 1) / (2 * l2 + 1))
    expected = sign * factor * jax.numpy.swapaxes(exchanged[:, ::-1, ::-1], 1, 2)

    assert jax.numpy.allclose(direct, expected, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_satisfies_the_lowering_recursion(seed):
    l1, l2, l3 = _random_triple(jax.random.key(seed))

    padded = jax.numpy.pad(clebsch_gordan_tensor(l1, l2, l3), 1)
    m1 = jax.numpy.arange(-l1, l1 + 1)[:, None, None]
    m2 = jax.numpy.arange(-l2, l2 + 1)[None, :, None]
    m3 = jax.numpy.arange(-l3, l3 + 1)[None, None, :]

    lowered_m3 = jax.numpy.sqrt((l3 + m3) * (l3 - m3 + 1)) * padded[1:2 * l1 + 2, 1:2 * l2 + 2, 0:2 * l3 + 1]
    lowered_m1 = jax.numpy.sqrt((l1 - m1) * (l1 + m1 + 1)) * padded[2:2 * l1 + 3, 1:2 * l2 + 2, 1:2 * l3 + 2]
    lowered_m2 = jax.numpy.sqrt((l2 - m2) * (l2 + m2 + 1)) * padded[1:2 * l1 + 2, 2:2 * l2 + 3, 1:2 * l3 + 2]

    assert jax.numpy.allclose(lowered_m3, lowered_m1 + lowered_m2, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_block_diagonalises_the_kronecker_product(seed):
    key_pair, key_rotation = jax.random.split(jax.random.key(seed), 2)
    l1, l2 = _random_pair(key_pair)
    rotation_vector = jax.random.normal(key_rotation, (3,))
    dimension = (2 * l1 + 1) * (2 * l2 + 1)
    l3_values = tuple(range(abs(l1 - l2), l1 + l2 + 1))

    basis = jax.numpy.concatenate(
        [clebsch_gordan_tensor(l1, l2, l3).reshape(dimension, -1) for l3 in l3_values],
        axis=1,
    )
    kronecker = jax.numpy.kron(
        wigner_D_tensor(l1, rotation_vector), wigner_D_tensor(l2, rotation_vector)
    )
    block_diagonal = jax.scipy.linalg.block_diag(
        *[wigner_D_tensor(l3, rotation_vector) for l3 in l3_values]
    )

    assert jax.numpy.allclose(basis.T @ kronecker @ basis, block_diagonal, atol=ATOL)


@pytest.mark.parametrize("seed", range(N_TEST))
def test_clebsch_gordan_tensor_is_an_intertwiner(seed):
    key_triple, key_rotation = jax.random.split(jax.random.key(seed), 2)
    l1, l2, l3 = _random_triple(key_triple)
    rotation_vector = jax.random.normal(key_rotation, (3,))

    clebsch_gordan = clebsch_gordan_tensor(l1, l2, l3)
    wigner_D_1 = wigner_D_tensor(l1, rotation_vector)
    wigner_D_2 = wigner_D_tensor(l2, rotation_vector)
    wigner_D_3 = wigner_D_tensor(l3, rotation_vector)

    coupled = jax.numpy.einsum("mM, nN, MNk -> mnk", wigner_D_1, wigner_D_2, clebsch_gordan)
    rotated = jax.numpy.einsum("mnK, Kk -> mnk", clebsch_gordan, wigner_D_3)

    assert jax.numpy.allclose(coupled, rotated, atol=ATOL)
