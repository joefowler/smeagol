# SMEAGOL: Spectrum Modeling, Emission, Attenuation, and Generation Optics Library for x-ray physics

A package to facilitate (?) modeling of x-ray fluorescence emission spectroscopy.

## Installation

```bash
cd ~/where/you_want_to/install
git clone git@github.com:joefowler/smeagol.git
cd smeagol
uv pip install -e .
```

Then from python, activities like this:

```python
import smeagol
from smeagol import Absorber, Emitter
```
