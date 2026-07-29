# Scrapped experiment: cubic-basis LSM for American options
#
# Wanted to see if a (1, x, x^2, x^3) basis beats the quadratic one.
# Result: barely different, but the matrix condition number got worse and the
# regression turned unstable (especially with few ITM paths). Not worth it,
# went back to quadratic. Kept as a note.

import numpy as np


def lsm_basis_cubic(X: np.ndarray) -> np.ndarray:
    # TODO: if I ever try higher orders, standardize first or the numerics blow up
    return np.column_stack([np.ones_like(X), X, X**2, X**3])
