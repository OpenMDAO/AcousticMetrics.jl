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

struct ExactProportionalBands{NO,TF}
    bstart::Int
    bend::Int
    function ExactProportionalBands{NO,TF}(bstart::Int, bend::Int) where {NO,TF}
        NO > 0 || throw(ArgumentError("Octave band fraction NO = $NO should be greater than 0"))
        bend > bstart || throw(ArgumentError("bend $bend should be greater than bstart $bstart"))
        return new{NO,TF}(bstart, bend)
    end
end

ExactProportionalBands{NO}(TF, bstart::Int, bend::Int) where {NO} = ExactProportionalBands{NO,TF}(bstart, bend)
ExactProportionalBands{NO}(bstart::Int, bend::Int) where {NO} = ExactProportionalBands{NO}(Float64, bstart, bend)

# Get the range of exact center bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactProportionalBands{NO}(fstart::TF, fend::TF) where {NO,TF} = ExactProportionalBands{NO,TF}(band_exact_lower(NO, fstart), band_exact_upper(NO, fend))

lower_bands(bands::ExactProportionalBands{NO,TF}) where {NO,TF} = ExactLowerBands{NO,TF}(bands.bstart, bands.bend)
center_bands(bands::ExactProportionalBands{NO,TF}) where {NO,TF} = ExactCenterBands{NO,TF}(bands.bstart, bands.bend)
upper_bands(bands::ExactProportionalBands{NO,TF}) where {NO,TF} = ExactUpperBands{NO,TF}(bands.bstart, bands.bend)

struct ExactProportionalBandSpectrumNB{NO,TF,Tpsd<:AbstractPowerSpectralDensity} <: AbstractVector{TF}
    bands::ExactProportionalBands{NO,TF}
    psd::Tpsd
end

function ExactProportionalBandSpectrumNB{NO}(psd::AbstractPowerSpectralDensity) where {NO}
    f = frequency(psd)
    # First frequency is always zero, and the frequencies are always evenly spaced, so the second frequency is the same as the spacing.
    Δf = f[2]
    # We're thinking of each non-zero freqeuncy as being a bin with center
    # frequency `f` and width `Δf`. So to get the lowest non-zero frequency
    # we'll subtract 0.5*Δf from the lowest non-zero frequency center:
    #   fstart = f[2] - 0.5*Δf = f[2] - 0.5(f[2]) = 0.5*f[2] = 0.5*Δf
    fstart = 0.5*Δf
    fend = last(f) + Δf

    bands = ExactProportionalBands{NO}(fstart, fend)

    return ExactProportionalBandSpectrumNB(bands, psd)
end

@inline function Base.size(pbs::ExactProportionalBandSpectrumNB)
    return (pbs.bands.bend - pbs.bands.bstart + 1,)
end

@inline Base.eltype(::Type{ExactProportionalBandSpectrumNB{NO,TF}}) where {NO,TF}= TF

@inline function Base.getindex(pbs::ExactProportionalBandSpectrumNB, i::Int)
    @boundscheck checkbounds(pbs, i)
    # This is where the fun begins.
    # So, first I want the lower and upper bands of this band.
    fl = lower_bands(pbs.bands)[i]
    fu = upper_bands(pbs.bands)[i]
    # Now I need to find the starting and ending indices that are included in
    # this frequency band.

    # Need the narrowband frequencies.
    f_nb = frequency(pbs.psd)

    # This is the narrowband frequency spacing.
    Δf = f_nb[2]

    # So, what is the first index we want?
    # It's the one that has f_nb[i] + 0.5*Δf >= fl.
    # So that means f_nb[i] >= fl - 0.5*Δf
    istart = searchsortedfirst(f_nb, fl - 0.5*Δf)
    # `searchsortedfirst` will return `length(f_nb)+1` it doesn't find anything.
    # What does that mean?
    # That means that all the frequencies in the narrowband spectrum are lower
    # than the band we're looking at. So return 0.
    if istart == length(f_nb) + 1
        return zero(eltype(pbs))
    end

    # What is the last index we want?
    # It's the last one that has f_nb[i] - 0.5*Δf <= fu
    # Or f_nb[i] <= fu + 0.5*Δf
    iend = searchsortedlast(f_nb, fu + 0.5*Δf)
    if iend == 0
        # All the frequencies are lower than the band we're looking for.
        return zero(eltype(pds))
    end

    # Need the psd amplitude relavent for this band.
    # Not sure if I can do that.
    psd_amp = amplitude(pbs.psd)
    psd_amp_v = @view psd_amp[istart:iend]
    f_nb_v = @view f_nb[istart:iend]
    
    # Get the contribution of the first band, which might not be a full band.
    # So the band will start at fl, the lower edge of the proportional band, and
    # end at the narrowband center frequency + 0.5*Δf.
    res_first_band = psd_amp_v[1]*(min((f_nb_v[1] + 0.5*Δf) - fl, Δf))

    # Get the contribution of the last band, which might not be a full band.
    res_last_band = psd_amp_v[end]*(min(fu - (f_nb_v[end] - 0.5*Δf), Δf))

    # Get all the others and return them.
    return res_first_band + sum(psd_amp_v[2:end-1]*Δf) + res_last_band
end
