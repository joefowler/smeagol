import numpy as np
import xraydb
from dataclasses import dataclass
from numpy.typing import NDArray


@dataclass
class Absorber:
    materials: list[str]
    thickness_cm: list[float]

    def attenuation(self, energy: NDArray | float) -> NDArray | float:
        return 1 - self.transmission(energy)

    def transmission(self, energy: NDArray | float) -> NDArray | float:
        transmitted = np.ones_like(energy, dtype=float)
        for material, thickness in zip(self.materials, self.thickness_cm):
            mu = xraydb.material_mu(material, energy)
            transmitted *= np.exp(-thickness * mu)
        return transmitted
