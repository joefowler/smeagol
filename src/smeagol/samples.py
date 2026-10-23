import numpy as np
import pylab as plt
import xraydb
from dataclasses import dataclass
from numpy.typing import NDArray


def is_element_symbol(s: str) -> bool:
    try:
        z = xraydb.atomic_number(s)
        return z is not None
    except Exception:
        return False


@dataclass(frozen=True)
class Sample:
    materials: list[str]
    mass_fraction: list[float | None]
    thickness_cm: float
    density: float
    elements: list[str]
    atomic_numbers: list[int]
    element_fraction: list[float]

    @classmethod
    def create(cls, materials: list[str], mass_fraction: list[float | None], thickness_cm: float, density: float):
        total_frac = sum(x for x in mass_fraction if x is not None)
        assert total_frac <= 1.0, f"the mass fractions sum to {total_frac}, must be <= 1"

        none_count = mass_fraction.count(None)
        assert none_count <= 1, "only one material can have a mass_fraction of None"
        if none_count == 1:
            missing_frac = 1.0 - total_frac
            for i, f in enumerate(mass_fraction):
                if f is None:
                    mass_fraction[i] = missing_frac
                    break

        elements: list[str] = []
        element_fraction: list[float] = []
        for material, mfrac in zip(materials, mass_fraction):
            assert mfrac is not None
            if is_element_symbol(material):
                elements.append(material)
                element_fraction.append(mfrac)
            else:
                try:
                    d = xraydb.chemparse(material)
                    total_mass = 0.0
                    for element, count in d.items():
                        total_mass += count * xraydb.atomic_mass(element)
                    for element, count in d.items():
                        element_mf = count * xraydb.atomic_mass(element) / total_mass
                        elements.append(element)
                        element_fraction.append(element_mf * mfrac)

                except ValueError:
                    assert material is not None
                    dummy_energy = 1e4
                    components = xraydb.material_mu_components(material, dummy_energy)
                    total_mass = components["mass"]
                    for element in components["elements"]:
                        count, atomic_mass, _ = components[element]
                        element_mf = count * atomic_mass / total_mass
                        elements.append(element)
                        element_fraction.append(element_mf * mfrac)

        atomic_numbers = [xraydb.atomic_number(s) for s in elements]
        return cls(materials, mass_fraction, thickness_cm, density, elements, atomic_numbers, element_fraction)

    def attenuation(self, energy: NDArray | float) -> NDArray | float:
        return 1 - self.transmission(energy)

    def transmission(self, energy: NDArray | float) -> NDArray | float:
        mu_total = np.zeros_like(energy, dtype=float)
        for material, mass_frac in zip(self.materials, self.mass_fraction):
            assert mass_frac is not None
            partial_density = self.density * mass_frac
            mu = xraydb.material_mu(material, energy, partial_density)
            mu_total += mu
        transmitted = np.exp(-mu_total * self.thickness_cm)
        return transmitted

    def plot_mu_components(self, energy: NDArray, proportion: bool = False) -> None:
        Tmu = self.total_loss_rate(energy, photo_only=False)

        plt.clf()
        for e, a, frac in zip(self.elements, self.atomic_numbers, self.element_fraction):
            mu_photo = xraydb.mu_elam(e, energy, "photo") * frac * self.density
            for shell in ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5"):
                edge = xraydb.xray_edge(e, shell)
                if edge is None or edge.jump_ratio == 1 or np.all(energy < edge.energy):
                    continue
                jr = edge.jump_ratio
                use = energy > edge.energy
                label = f"{e} (Z={a} {shell})"
                this_mu = (jr - 1) / jr * mu_photo[use]
                y = this_mu
                if proportion:
                    y = 100 * y / Tmu[use]
                plt.plot(energy[use], y, label=label)
                mu_photo[use] -= this_mu
        plt.legend()
        plt.xlabel("Energy (eV)")
        if proportion:
            Tphoto = self.total_loss_rate(energy, photo_only=True)
            plt.plot(energy, 100 * Tphoto / Tmu, "k", lw=2)
            plt.ylabel("Percent of all events at that energy (%)")
            plt.title("Percent of absorptions that cause excitation")
        else:
            plt.ylabel("$\\mu$ (cm$^{-1}$)")
            plt.title("Attenuation rate (per cm) for all components of sample")
            plt.plot(energy, Tmu, "k", lw=2)
            plt.semilogy()

    def total_loss_rate(self, energy: NDArray, photo_only: bool = False) -> NDArray:
        kind = "total" if photo_only else "photo"
        Tmu = np.zeros_like(energy)
        for e, frac in zip(self.elements, self.element_fraction):
            mu_total = xraydb.mu_elam(e, energy, kind=kind) * frac * self.density
            Tmu += mu_total
        return Tmu
