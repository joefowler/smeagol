import numpy as np

# import pylab as plt
import scipy as sp
import xraydb
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Emitter:
    element: str | int
    K_only: bool = True

    def Kemission(self, energy: NDArray, intensity: NDArray) -> float:
        """Total K-line emission given the source intensity spectrum"""
        edge = xraydb.xray_edge(self.element, "K")
        assert edge is not None
        if np.max(energy) < edge.energy:
            return 0.0
        use = energy > edge.energy
        fyield = edge.fyield
        jr = edge.jump_ratio
        k_fraction = (jr - 1) / jr

        symbol = xraydb.atomic_symbol(self.element)
        density = xraydb.atomic_density(self.element)
        mu_photo = xraydb.material_mu(symbol, energy[use], kind="photo", density=density)

        # Assume an optically thin material
        # TODO: replace this placeholder for the emitter thickness
        thickness = 1.0e-4
        k_shell_generation = mu_photo * k_fraction * thickness * fyield

        total_k_emission = sp.integrate.simpson(intensity[use] * k_shell_generation, energy[use])
        return total_k_emission

    def Kspectrum(self, energy: NDArray) -> NDArray:
        s = np.zeros_like(energy, dtype=float)
        Kwidth = float(xraydb.core_width(self.element, "K"))  # type: ignore
        for _linename, line in xraydb.xray_lines(self.element, "K").items():
            lineE = line.energy
            final_level = line.final_level
            if "," in final_level:
                final_level = final_level.split(",")[0]
            width = Kwidth + float(xraydb.core_width(self.element, final_level))  # type: ignore
            gamma = width * 0.5
            s += line.intensity / (np.pi * gamma * (1 + ((energy - lineE) / gamma) ** 2))
        return s
