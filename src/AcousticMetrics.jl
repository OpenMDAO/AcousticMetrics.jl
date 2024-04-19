module AcousticMetrics

using Base.Iterators: Iterators
using Base.Order: ord, Forward
using ConcreteStructs: @concrete
using FFTW: r2r!, R2HC, HC2R, rfftfreq
using FLOWMath: abs_cs_safe
using ForwardDiff: ForwardDiff
using OffsetArrays: OffsetArray

include("constants.jl")
include("fourier_transforms.jl")
include("narrowband.jl")
export AbstractPressureTimeHistory, PressureTimeHistory

include("proportional_bands.jl")

include("weighting.jl")

end # module
