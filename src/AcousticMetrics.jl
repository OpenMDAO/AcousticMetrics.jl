module AcousticMetrics

using ConcreteStructs: @concrete
using FFTW: r2r!, R2HC
using ForwardDiff: ForwardDiff
using OffsetArrays: OffsetArray

include("constants.jl")
export p_ref

include("weighting.jl")
export W_A

include("fourier_transforms.jl")
export RFFTCache, rfft!, rfft, rfftfreq

include("metrics.jl")
export nbs_from_apth, oaspl_from_apth, oaspl_from_nbs

end # module
