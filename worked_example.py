"""The worked example in Section 3 of the note: constant group-specific reporting changes the
off-diagonal entries and the settled mix, but not the spectral radius (the long-run R)."""
import numpy as np

from common import spectral_radius, settled_mix

p = np.array([1.0, 0.5])
C = np.array([[2, 4.8], [1.2, 3.84]])
q = np.array([0.2, 0.6])
M = p[:, None] * C                       # diag(p) C, rows = infectors
Mt = np.diag(1 / q) @ M @ np.diag(q)     # matrix followed by reported cases, Q^-1 M Q

for name, A in [("M (infections)", M), ("Q^-1 M Q (reported cases)", Mt)]:
    print(f"{name}:\n{A}")
    print(f"  trace {np.trace(A):.2f}, determinant {np.linalg.det(A):.2f}, "
          f"spectral radius {spectral_radius(A):.3f}, settled share in group 1 {settled_mix(A)[0]:.2f}\n")
