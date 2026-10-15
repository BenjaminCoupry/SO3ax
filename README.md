# SO3ax

A small JAX package for the representation theory of SO(3): the coefficient
tensors that rotation-equivariant models are built from. It provides
Clebsch-Gordan coupling tensors, `so(3)` Lie algebra generators, and Wigner
D-matrices, each in the complex `|l, m>` basis and in a real spherical harmonic
basis. It computes the tensors and leaves you to contract them into your own
model.

## Installation

Clone the repository and install the package in editable mode:

```
git clone <repository-url>
cd SO3ax
python -m pip install -e .
```

The examples and the test suite evaluate the spherical harmonics themselves
with `sphericart`, pulled in by the `test` extra:

```
python -m pip install -e ".[test]"
```

## Examples

### Rotating spherical harmonics

```python
import jax
import sphericart.jax
from so3ax import so3_to_log_coordinates, wigner_D_real_tensor

lmax = 4
rotation = jax.random.orthogonal(jax.random.key(0), 3)
rotation = rotation.at[:, 0].multiply(jax.numpy.linalg.det(rotation))  # into SO(3)
points = jax.random.normal(jax.random.key(1), (256, 3))

harmonics = sphericart.jax.spherical_harmonics(points, lmax)
rotation_vector = so3_to_log_coordinates(rotation)
wigner_D = jax.scipy.linalg.block_diag(
    *[wigner_D_real_tensor(l, rotation_vector) for l in range(lmax + 1)]
)

# rotating the points and rotating their spherical harmonics agree
assert jax.numpy.allclose(
    sphericart.jax.spherical_harmonics(points @ rotation.T, lmax),
    harmonics @ wigner_D.T,
    atol=1e-4,  # float32 precision; much tighter with float64
)
```

### Coupling two features

```python
import jax
import sphericart.jax
from so3ax import clebsch_gordan_real_tensor, so3_to_log_coordinates, wigner_D_real_tensor

l1, l2, l3 = 1, 2, 3
coupling = clebsch_gordan_real_tensor(l1, l2, l3)  # (3, 5, 7)

rotation = jax.random.orthogonal(jax.random.key(0), 3)
rotation = rotation.at[:, 0].multiply(jax.numpy.linalg.det(rotation))
point = jax.random.normal(jax.random.key(1), (3,))

def coupled_feature(x):
    feature_1 = sphericart.jax.spherical_harmonics(x, l1)[l1 * l1:]
    feature_2 = sphericart.jax.spherical_harmonics(x, l2)[l2 * l2:]
    return jax.numpy.einsum("a, b, abc -> c", feature_1, feature_2, coupling)

wigner_D = wigner_D_real_tensor(l3, so3_to_log_coordinates(rotation))

# coupling then rotating is the same as rotating then coupling
assert jax.numpy.allclose(
    wigner_D @ coupled_feature(point),
    coupled_feature(rotation @ point),
    atol=1e-6,
)
```

## Tensors

Every quantity comes in two bases. `clebsch_gordan_tensor`, `generators_tensor`
and `wigner_D_tensor` act on the complex `|l, m>` basis, ordered `m = -l, ..., l`.
Their `_real_tensor` counterparts act on the real spherical harmonic basis, so
`wigner_D_real_tensor` rotates a real harmonic vector directly and
`generators_real_tensor` returns real antisymmetric matrices. `real_to_complex(l)`
is the unitary matrix relating the two.

Rotations are passed as logarithmic coordinates: a rotation vector whose
direction is the axis and whose norm is the angle. `so3_to_log_coordinates`
recovers them from a rotation matrix (not differentiable at a half-turn).

## Precision

SO3ax does not change JAX's precision settings: tensors come back in JAX's
default floating dtype, which is float32 unless 64-bit mode is enabled. That is
accurate for low degrees, but the error grows quickly with `l`. 
For high degrees, enable float64 before any JAX computation:

```python
import jax

jax.config.update("jax_enable_x64", True)  # or run with JAX_ENABLE_X64=1

import so3ax
```

To get float64 tensors without switching the rest of the program to 64-bit,
compute them under the context manager:

```python
import numpy

with jax.enable_x64(True):
    coupling =so3ax.clebsch_gordan_real_tensor(l1, l2, l3)
```

The test suite scales its tolerance and its largest `l` with the default dtype;
run `JAX_ENABLE_X64=1 python -m pytest` to test at full precision.

## Related libraries

The real spherical harmonics are evaluated with
[sphericart](https://github.com/lab-cosmo/sphericart); `wigner_D_real_tensor` and
`clebsch_gordan_real_tensor` act on its output directly, with no coordinate
reordering.

The conventions follow [e3nn-jax](https://github.com/e3nn/e3nn-jax), rewritten
from scratch in plain JAX, up to two differences:

- **Normalisation.** `clebsch_gordan_real_tensor(l1, l2, l3)` is
  `sqrt(2 * l3 + 1)` times e3nn-jax's `so3._clebsch_gordan(l1, l2, l3)` (the
  physicists' Clebsch-Gordan coefficient rather than e3nn-jax's
  `1 / sqrt(2 * l3 + 1)` scaling).
- **Axes.** A rotation vector `(wx, wy, wz)` here is `(wy, -wz, -wx)` for
  e3nn-jax's `su2_generators`.

## References

- Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). *Quantum Theory of Angular Momentum*. World Scientific.
- Racah, G. (1942). Theory of Complex Spectra. II. *Physical Review*, 62, 438-462.
- Bigi, F., Fraux, G., Browning, N. J., & Ceriotti, M. (2023). Fast evaluation of spherical harmonics with sphericart. *The Journal of Chemical Physics*, 159(6), 064802.
- Geiger, M., & Smidt, T. (2022). *e3nn: Euclidean Neural Networks*. arXiv:2207.09453.
