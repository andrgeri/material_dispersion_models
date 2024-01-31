# Material dispersion models

Includes:
 * AlGaAs
 * AlGaInP lattice-matched to GaAs
 * SiO2
 * TiO2
 * GaN
 * GaP
 * Electronic resists

Generally, the exported function can be called as
```python

import numpy as np

wl = np.linspace(0.5, 2.0) * 1e-6  # [m], wavelength
n = foo(wl)

```
