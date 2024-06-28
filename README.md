[![Tests](https://github.com/OpenMDAO/AcousticMetrics.jl/actions/workflows/test.yaml/badge.svg)](https://github.com/OpenMDAO/AcousticMetrics.jl/actions/workflows/test.yaml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://OpenMDAO.github.io/AcousticMetrics.jl/dev)

# Introduction
AcousticMetrics.jl is a Julia package for computing various metrics useful in acoustics.
Currently implemented metrics include:

  * Various narrowband spectra

    * Pressure amplitude
    * Mean-squared pressure amplitude (MSP)
    * Power Spectral Density (PSD)
    * Phase

  * Proportional band spectra

    * Approximate octave and third-octave spectra
    * Exact proportional spectra of any octave fraction > 0.
    * Lazy representations of proportional band spectra constructed from either other narrowband or proportional band spectra

  * Integrated metrics

    * Unweighted and A-weighted Overall Sound Pressure Level (OASPL)
