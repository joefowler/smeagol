"""
SMEAGOL: Spectrum Modeling, Emission, Attenuation, and Generation Object Library.
"""

__version__ = "0.1.0"

# Import core classes to the top level of the package namespace
from .absorbers import Absorber
from .emitters import Emitter
from .samples import Sample
# from .source import TubeSource
# from .filter import Filter
# from .detector import Detector
# from .system import Beamline

# Define what gets imported when a user runs `from smeagol import *`
__all__ = [
    "Absorber",
    "Emitter",
    "Sample",
    # "TubeSource",
    # "Filter",
    # "Detector",
    # "Beamline",
]
