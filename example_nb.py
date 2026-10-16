import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pylab as plt
    from numpy.typing import NDArray
    from smeagol import Absorber, Emitter


    return Absorber, Emitter, NDArray, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Source Spectrum
    Let's assume a very crude source model: a Gaussian peaking at 52 keV and with a FWHM of 16 keV (44 to 60 keV) and with a max E of 70 keV. This is based on a specific example spectrum provided by Nathan. We'll assume that his scattering background is similar in shape to the source spectrum. For now, the peak intensity is undefined, or in undefined units.

    For the background spectrum, we'll assume the same, with a peak intensity of 40 cps per 40 eV bin, except that there's also a minimum of 1 cps per bin.
    """)
    return


@app.cell
def _(np, plt):
    e = np.linspace(1000, 70000, 7001)

    ctr = 52e3
    fwhm = 16e3
    sigma = fwhm / 2.3548
    source = 40 * np.exp(-0.5 * ((e-ctr)*1.5/sigma)**2)
    background = 40 * np.exp(-0.5 * ((e-ctr)/sigma)**2)
    bmin = 1.5
    background += bmin
    background += 60 * np.exp(-0.5 * (e/6000)**2)

    plt.clf()
    plt.plot(e/1e3, background)
    plt.plot(e/1e3, source)
    plt.ylim(.3, 100)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (cps per 1 eV bin)")
    plt.semilogy()
    plt.grid()
    plt.gca()
    return background, e, source


@app.cell
def _(mo):
    mo.md(r"""
    ## Filtering
    Assume that any high-pass filter material built into the source is already accounted for by having the Gaussian-shaped source spectrum. This is more expedient than combining a bremsstrahlung ([Kramers law](https://en.wikipedia.org/wiki/Kramers'_law)) spectrum with a filter material to block the lower-energy photons. BUT our sample is contained in a sample chamber that we _do_ want to account for.

    We'll assume it's PTFE (Teflon) of 1 mm thickness.
    """)
    return


@app.cell
def _(Absorber, e, plt, source):

    chamber = Absorber(["Teflon"], [0.1])
    source_at_sample = source * chamber.transmission(e)
    plt.clf()
    plt.plot(e/1e3, chamber.transmission(e))
    plt.xlabel("Energy (keV)")
    plt.ylabel("Transmitted fraction")
    plt.title("Transmission through 1 mm Teflon")
    plt.grid()
    plt.gca()
    return chamber, source_at_sample


@app.cell
def _(mo):
    ppm_amounts = {
        "Se": 145,
        "Sr": 34,
        "Zr": 30,
        "Nb": 75,
        "Mo": 234,
        "Ag": 310,
        "Cd": 280,
        "Sn": 130,
        "Sb": 200,
        "Ba": 110,
        # "W": 45,
        # "Au": 20,
        # "Hg": 6,
        # "Pb": 185,
        # "Th": 165,
        # "U": 28,
    }
    mo.md("""## Sample elements
    Have a few 100s of ppm of several elements""")
    return (ppm_amounts,)


@app.cell
def _(Emitter, e, ppm_amounts, source_at_sample):
    scale_per_ppm = {}
    for sample, ppm in ppm_amounts.items():
        xrfsource = Emitter(sample)
        scale = xrfsource.Kemission(e, source_at_sample)
        scale_per_ppm[sample] = scale
        print(f"{sample:2s} {ppm:3d} ppm {scale:6.2f} photons per ?? source per ppm")
    return (scale_per_ppm,)


@app.cell
def _(NDArray, np):
    def smear(energy: NDArray, spectrum: NDArray, resolution: float | NDArray) -> NDArray:
        N = energy.shape[0]
        assert N == spectrum.shape[0]
    
        if isinstance(resolution, float):
            resolution = np.full(N, resolution)
        smeared = np.zeros((N, ))
        de = energy[1] - energy[0]
        for i, e in enumerate(energy):
            r = resolution[i]
            sigma = r / 2.3548
            smear_row = np.exp(-0.5 * ((energy - e)/sigma)**2) / (2 * np.pi * sigma) * de
            smeared += smear_row * spectrum[i]
        return smeared

    return (smear,)


@app.cell
def _(Emitter, chamber, e, ppm_amounts):
    # Compute the raw spectrum for each element
    raw_spectra = {}
    for _sample in ppm_amounts:
        xrfsource1 = Emitter(_sample)
        raw_spectra[_sample] = xrfsource1.Kspectrum(e) * chamber.transmission(e)

    return (raw_spectra,)


@app.cell
def _(
    Absorber,
    background,
    e,
    mo,
    np,
    plt,
    ppm_amounts,
    raw_spectra,
    scale_per_ppm,
    smear,
):
    overall_spectrum = np.zeros_like(e, dtype=float)
    overall_spectrum = background.copy()
    resolution_vs_e = (40**2 + 2.4 * e) ** 0.5

    GLOBAL_SCALE = 1.
    Amptek = Absorber(["Si"], [0.1])  # assume it's the 1000 µm version of the Amptek FAST SDD

    plt.clf()
    for _sample in ppm_amounts:
        ppm1 = ppm_amounts[_sample]
        scale1 = scale_per_ppm[_sample]
    
        s = raw_spectra[_sample] * ppm1 * scale1 * GLOBAL_SCALE
        s *= Amptek.attenuation(e)
        s = smear(e, s, resolution_vs_e)
        overall_spectrum += s
        plt.plot(e/1e3, s, label=_sample)
    plt.plot(e/1e3, overall_spectrum, "k", lw=2.5)
    plt.xlim(10, 45)
    plt.ylim(.5, 700)
    plt.semilogy()
    plt.legend(loc="upper right")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (cps)ll")
    plt.title("Expected spectrum at Amptek")
    plt.grid()
    # plt.gca()l
    mo.mpl.interactive(plt.gca())
    return (GLOBAL_SCALE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Now a Rare Earth sample ...?
    """)
    return


@app.cell
def _(Emitter, chamber, e, raw_spectra, scale_per_ppm, source_at_sample):
    ppm_rareearth = {
        "Ba": 500,
        "La": 500,
        "Ce": 333,
        "Pr": 433,
        "Nd": 300,
        "Sm": 900,
        "Gd": 360,
        "Y": 400,
    }
    for _sample, _ppm in ppm_rareearth.items():
        _xrfsource = Emitter(_sample)
        _scale = _xrfsource.Kemission(e, source_at_sample)
        scale_per_ppm[_sample] = _scale
        print(f"{_sample:2s} {_ppm:3d} ppm {_scale:7.2f} photons per ?? source per ppm")
        raw_spectra[_sample] = _xrfsource.Kspectrum(e) * chamber.transmission(e)
    
    return (ppm_rareearth,)


@app.cell
def _(
    Absorber,
    GLOBAL_SCALE,
    background,
    e,
    mo,
    plt,
    ppm_rareearth,
    raw_spectra,
    scale_per_ppm,
    smear,
):
    def TES_result():
        # overall_spectrum = np.zeros_like(e, dtype=float)
        overall_spectrum = background.copy()
        resolution_vs_e = 20.0  # start with a 20 eV resolution
        TES = Absorber(["Bi", "Au"], [25e-4, 2e-4])  # Joel W said ~20-30 µm max bismuth. Add 2 µm gold
        plt.clf()
        for _sample in ppm_rareearth:
            ppm1 = ppm_rareearth[_sample]
            scale1 = scale_per_ppm[_sample]
        
            s = raw_spectra[_sample] * ppm1 * scale1 * GLOBAL_SCALE
            s *= TES.attenuation(e)
            s = smear(e, s, resolution_vs_e)
            overall_spectrum += s
            plt.plot(e/1e3, s, label=_sample)
        plt.plot(e/1e3, overall_spectrum, "k", lw=2.5)
        plt.xlim(10, 55)
        plt.ylim(.5, 7000)
        plt.semilogy()
        plt.legend(loc="upper right")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Intensity (cps)ll")
        plt.title("Expected spectrum at TES array ($\\delta E=25$ eV)")
        plt.grid()
        return overall_spectrum
        # plt.gca()l
    spectrum1 = TES_result()
    mo.mpl.interactive(plt.gca())
    return (spectrum1,)


@app.cell
def _(
    Absorber,
    GLOBAL_SCALE,
    background,
    e,
    mo,
    plt,
    ppm_rareearth,
    raw_spectra,
    scale_per_ppm,
    smear,
):
    def Amptek_result():
        # overall_spectrum = np.zeros_like(e, dtype=float)
        overall_spectrum = background.copy()
        resolution_vs_e = (40**2 + 2.4 * e) ** 0.5
        res50 = (40**2 + 2.4*5e4) ** 0.5
        Amptek = Absorber(["Si"], [0.1])  # assume it's the 1000 µm version of the Amptek FAST SDD
    
        plt.clf()
        for _sample in ppm_rareearth:
            ppm1 = ppm_rareearth[_sample]
            scale1 = scale_per_ppm[_sample]
        
            s = raw_spectra[_sample] * ppm1 * scale1 * GLOBAL_SCALE
            s *= Amptek.attenuation(e)
            s = smear(e, s, resolution_vs_e)
            overall_spectrum += s
            plt.plot(e/1e3, s, label=_sample)
        plt.plot(e/1e3, overall_spectrum, "k", lw=2.5)
        plt.xlim(10, 55)
        plt.ylim(.5, 7000)
        plt.semilogy()
        plt.legend(loc="upper right")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Intensity (cps)ll")
        plt.title(f"Expected spectrum at TES array ($\\delta E={res50:.0f}$ eV at 50 keV, 40 eV at 0 eV)")
        plt.grid()
        return overall_spectrum
        # plt.gca()l
    spectrum2 = Amptek_result()
    mo.mpl.interactive(plt.gca())
    return (spectrum2,)


@app.cell
def _(mo):
    mo.md(r"""
    Notice how the Ce K$\beta$ lines (green) influence the ability to isolate the Sm K$\alpha$ lines (brown). Also the barium (blue) interferes with the neodymium (purple). And Nd messes with the Gd (purple under ping). This happens all over the spectrum, to some degree.
    """)
    return


@app.cell
def _(e, mo, plt, spectrum1, spectrum2):
    plt.clf()
    plt.plot(e/1e3, spectrum1, label="20 eV TES")
    plt.plot(e/1e3, spectrum2, label="300 eV SDD")
    plt.xlim(10, 55)
    plt.ylim(.5, 7000)
    plt.xlabel("Energy (keV)")
    plt.grid()
    plt.legend(loc="upper left")
    plt.semilogy()
    plt.title("TES vs SDD spectrum")
    mo.mpl.interactive(plt.gca())
    return


if __name__ == "__main__":
    app.run()
