```@meta
CurrentModule = AMDocs
```
# API Reference

## Fourier Transforms
```@docs
AcousticMetrics.rfft!
AcousticMetrics.irfft!
```

## Pressure Time History
```@docs
AbstractPressureTimeHistory
PressureTimeHistory
AcousticMetrics.pressure
AcousticMetrics.inputlength
AcousticMetrics.timestep(pth::AbstractPressureTimeHistory)
AcousticMetrics.starttime(pth::AbstractPressureTimeHistory)
AcousticMetrics.time
```

## Narrowband Metrics
```@docs
AbstractNarrowbandSpectrum
AcousticMetrics.halfcomplex
AcousticMetrics.timestep(sm::AbstractNarrowbandSpectrum)
AcousticMetrics.starttime(sm::AbstractNarrowbandSpectrum)
AcousticMetrics.samplerate
AcousticMetrics.frequency
AcousticMetrics.frequencystep
AcousticMetrics.istonal
PressureSpectrumAmplitude
PressureSpectrumPhase
MSPSpectrumAmplitude
MSPSpectrumPhase
PowerSpectralDensityAmplitude
PowerSpectralDensityPhase
```

### Proportional Bands and Proportional Band Spectra
```@docs
AbstractProportionalBands
AcousticMetrics.octave_fraction
AcousticMetrics.lower_center_upper
AcousticMetrics.freq_scaler
AcousticMetrics.band_start
AcousticMetrics.band_end
AcousticMetrics.lower_bands
AcousticMetrics.upper_bands
AcousticMetrics.center_bands
AcousticMetrics.cband_number
ExactProportionalBands
ExactOctaveCenterBands
ExactOctaveLowerBands
ExactOctaveUpperBands
ExactThirdOctaveCenterBands
ExactThirdOctaveLowerBands
ExactThirdOctaveUpperBands
ApproximateThirdOctaveBands
ApproximateThirdOctaveCenterBands
ApproximateThirdOctaveLowerBands
ApproximateThirdOctaveUpperBands
ApproximateOctaveBands
ApproximateOctaveCenterBands
ApproximateOctaveLowerBands
ApproximateOctaveUpperBands
AbstractProportionalBandSpectrum
AcousticMetrics.has_observer_time
AcousticMetrics.observer_time
AcousticMetrics.timestep(pbs::AbstractProportionalBandSpectrum)
AcousticMetrics.amplitude
AcousticMetrics.time_period
AcousticMetrics.time_scaler
LazyNBProportionalBandSpectrum
AcousticMetrics.frequency_nb
AcousticMetrics.lazy_pbs
ProportionalBandSpectrum
LazyPBSProportionalBandSpectrum
ProportionalBandSpectrumWithTime
AcousticMetrics.combine
```

## Weighting
```@docs
W_A
```

## Integrated Metrics
```@docs
OASPL
```
