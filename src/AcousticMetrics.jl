module AcousticMetrics

using Base.Iterators: Iterators
using Base.Order: ord, Forward
using FFTW: r2r!, R2HC, HC2R, rfftfreq
using FLOWMath: abs_cs_safe
using ForwardDiff: ForwardDiff

include("constants.jl")

include("fourier_transforms.jl")

include("narrowband.jl")
export AbstractPressureTimeHistory, PressureTimeHistory
export AbstractNarrowbandSpectrum
export PressureSpectrumAmplitude, PressureSpectrumPhase, MSPSpectrumAmplitude, MSPSpectrumPhase, PowerSpectralDensityAmplitude, PowerSpectralDensityPhase

include("proportional_bands.jl")
export AbstractProportionalBands
export ExactProportionalBands
export ExactOctaveCenterBands, ExactOctaveLowerBands, ExactOctaveUpperBands
export ExactThirdOctaveCenterBands, ExactThirdOctaveLowerBands, ExactThirdOctaveUpperBands
export ApproximateOctaveBands, ApproximateOctaveCenterBands, ApproximateOctaveLowerBands, ApproximateOctaveUpperBands
export ApproximateThirdOctaveBands, ApproximateThirdOctaveCenterBands, ApproximateThirdOctaveLowerBands, ApproximateThirdOctaveUpperBands
export AbstractProportionalBandSpectrum, LazyNBProportionalBandSpectrum, ProportionalBandSpectrum, ProportionalBandSpectrumWithTime, LazyPBSProportionalBandSpectrum

include("weighting.jl")

end # module
