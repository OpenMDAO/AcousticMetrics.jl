const f0_exact = 1000

@inline band_exact_lower(NO, fl) = floor(Int, 1/2 + NO*log2(fl/f0_exact) + 10*NO)
@inline band_exact_upper(NO, fu) = ceil(Int, -1/2 + NO*log2(fu/f0_exact) + 10*NO)

struct ExactCenterBands{NO,TF} <: AbstractVector{TF}
    bstart::Int
    bend::Int
    f0::TF
    function ExactCenterBands{NO,TF}(bstart::Int, bend::Int) where {NO,TF}
        NO > 0 || throw(ArgumentError("Octave band fraction NO = $NO should be greater than 0"))
        bend > bstart || throw(ArgumentError("bend $bend should be greater than bstart $bstart"))
        return new{NO,TF}(bstart, bend, TF(f0_exact))
    end
    function ExactCenterBands{NO}(TF, bstart::Int, bend::Int) where {NO}
        return ExactCenterBands{NO,TF}(bstart, bend)
    end
end

ExactCenterBands{NO}(bstart::Int, bend::Int) where {NO} = ExactCenterBands{NO}(Float64, bstart, bend)

# Get the range of exact center bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactCenterBands{NO}(fstart::TF, fend::TF) where {NO,TF} = ExactCenterBands{NO,TF}(band_exact_lower(NO, fstart), band_exact_upper(NO, fend))


@inline function Base.size(bands::ExactCenterBands)
    return (bands.bend - bands.bstart + 1,)
end

@inline function Base.getindex(bands::ExactCenterBands{NO}, i::Int) where {NO}
    @boundscheck checkbounds(bands, i)
    # Now, how do I get the band?
    # This is the band number:
    b = bands.bstart + (i - 1)
    # So then the center frequency f_c that I want is defined by the function
    # 
    #   b = NO*log2(f_c/f_0) + 10*NO
    # 
    # where f_0 is the reference frequency, 1000 Hz.
    # OK, so.
    #   2^((b - 10*NO)/NO)*f_c
    return 2^((b - 10*NO)/NO)*bands.f0
end

const ExactOctaveCenterBands{TF} = ExactCenterBands{1,TF}
const ExactThirdOctaveCenterBands{TF} = ExactCenterBands{3,TF}

struct ExactLowerBands{NO,TF} <: AbstractVector{TF}
    cbands::ExactCenterBands{NO,TF}
    cband2lband::TF
    function ExactLowerBands{NO,TF}(bstart::Int, bend::Int) where {NO,TF}
        cband2lband = TF(2)^(TF(-1)/(2*NO)) # Don't know if this is necessary.
        return new{NO,TF}(ExactCenterBands{NO,TF}(bstart, bend), cband2lband)
    end
    function ExactLowerBands{NO}(TF, bstart::Int, bend::Int) where {NO}
        return ExactLowerBands{NO,TF}(bstart, bend)
    end
end

ExactLowerBands{NO}(bstart::Int, bend::Int) where {NO} = ExactLowerBands{NO}(Float64, bstart, bend)

# Get the range of exact lower bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactLowerBands{NO}(fstart::TF, fend::TF) where {NO,TF} = ExactLowerBands{NO,TF}(band_exact_lower(NO, fstart), band_exact_upper(NO, fend))


@inline function Base.size(bands::ExactLowerBands)
    return size(bands.cbands)
end

@inline function Base.getindex(bands::ExactLowerBands, i::Int)
    @boundscheck checkbounds(bands.cbands, i)
    return bands.cbands[i]*bands.cband2lband
end

const ExactOctaveLowerBands{TF} = ExactLowerBands{1,TF}
const ExactThirdOctaveLowerBands{TF} = ExactLowerBands{3,TF}

struct ExactUpperBands{NO,TF} <: AbstractVector{TF}
    cbands::ExactCenterBands{NO,TF}
    cband2uband::TF
    function ExactUpperBands{NO,TF}(bstart::Int, bend::Int) where {NO,TF}
        cband2uband = TF(2)^(TF(1)/(2*NO)) # Don't know if all the `TF`s are necessary.
        return new{NO,TF}(ExactCenterBands{NO,TF}(bstart, bend), cband2uband)
    end
    function ExactUpperBands{NO}(TF, bstart::Int, bend::Int) where {NO}
        return ExactUpperBands{NO,TF}(bstart, bend)
    end
end

ExactUpperBands{NO}(bstart::Int, bend::Int) where {NO} = ExactUpperBands{NO}(Float64, bstart, bend)

# Get the range of third-octave upper bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactUpperBands{NO}(fstart::TF, fend::TF) where {NO,TF} = ExactUpperBands{NO,TF}(band_exact_lower(NO, fstart), band_exact_upper(NO, fend))

@inline function Base.size(bands::ExactUpperBands)
    return size(bands.cbands)
end

@inline function Base.getindex(bands::ExactUpperBands, i::Int)
    @boundscheck checkbounds(bands.cbands, i)
    return bands.cbands[i]*bands.cband2uband
end

const ExactOctaveUpperBands{TF} = ExactUpperBands{1,TF}
const ExactThirdOctaveUpperBands{TF} = ExactUpperBands{3,TF}
