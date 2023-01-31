abstract type AbstractProportionalBands{NO,LCU,TF} <: AbstractVector{TF} end

octave_fraction(::Type{<:AbstractProportionalBands{NO}}) where {NO} = NO
lower_center_upper(::Type{<:AbstractProportionalBands{NO,LCU,TF}}) where {NO,LCU,TF} = LCU

const f0_exact = 1000
const fmin_exact = 1

struct ExactProportionalBands{NO,LCU,TF} <: AbstractProportionalBands{NO,LCU,TF}
    bstart::Int
    bend::Int
    f0::TF
    function ExactProportionalBands{NO,LCU,TF}(bstart::Int, bend::Int) where {NO,LCU,TF}
        NO > 0 || throw(ArgumentError("Octave band fraction NO = $NO should be greater than 0"))
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU type must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        return new{NO,LCU,TF}(bstart, bend, TF(f0_exact))
    end
    function ExactProportionalBands{NO,LCU}(TF, bstart::Int, bend::Int) where {NO,LCU}
        return ExactProportionalBands{NO,LCU,TF}(bstart, bend)
    end
end

@inline band_start(bands::AbstractProportionalBands) = bands.bstart
@inline band_end(bands::AbstractProportionalBands) = bands.bend
@inline function Base.size(bands::AbstractProportionalBands)
    return (band_end(bands) - band_start(bands) + 1,)
end

function bands_lower(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF) where {NO,TF}
    return TBands{:lower}(fstart, fend)
end

function bands_upper(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF) where {NO,TF}
    return TBands{:upper}(fstart, fend)
end

function bands_center(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF) where {NO,TF}
    return TBands{:center}(fstart, fend)
end

ExactProportionalBands{NO,LCU}(bstart::Int, bend::Int) where {NO,LCU} = ExactProportionalBands{NO,LCU}(Float64, bstart, bend)

@inline band_exact_lower_limit(NO, fl) = floor(Int, 1/2 + NO*log2(fl/f0_exact) + 10*NO)
@inline band_exact_upper_limit(NO, fu) = ceil(Int, -1/2 + NO*log2(fu/f0_exact) + 10*NO)

# Get the range of exact center bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactProportionalBands{NO,LCU}(fstart::TF, fend::TF) where {NO,LCU,TF} = ExactProportionalBands{NO,LCU,TF}(fstart, fend)
ExactProportionalBands{NO,LCU,TF}(fstart::TF, fend::TF) where {NO,LCU,TF} = ExactProportionalBands{NO,LCU,TF}(band_exact_lower_limit(NO, fstart), band_exact_upper_limit(NO, fend))

@inline function Base.getindex(bands::ExactProportionalBands{NO,:center}, i::Int) where {NO}
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

@inline function Base.getindex(bands::ExactProportionalBands{NO,:lower}, i::Int) where {NO}
    @boundscheck checkbounds(bands, i)
    b = bands.bstart + (i - 1)
    # return 2^((b - 10*NO)/NO)*(2^(-1/(2*NO)))*bands.f0
    # return 2^(2*(b - 10*NO)/(2*NO))*(2^(-1/(2*NO)))*bands.f0
    return 2^((2*(b - 10*NO) - 1)/(2*NO))*bands.f0
end

@inline function Base.getindex(bands::ExactProportionalBands{NO,:upper}, i::Int) where {NO}
    @boundscheck checkbounds(bands, i)
    b = bands.bstart + (i - 1)
    # return 2^((b - 10*NO)/NO)*(2^(1/(2*NO)))*bands.f0
    # return 2^(2*(b - 10*NO)/(2*NO))*(2^(1/(2*NO)))*bands.f0
    return 2^((2*(b - 10*NO) + 1)/(2*NO))*bands.f0
end

const ExactOctaveCenterBands{TF} = ExactProportionalBands{1,:center,TF}
const ExactThirdOctaveCenterBands{TF} = ExactProportionalBands{3,:center,TF}
const ExactOctaveLowerBands{TF} = ExactProportionalBands{1,:lower,TF}
const ExactThirdOctaveLowerBands{TF} = ExactProportionalBands{3,:lower,TF}
const ExactOctaveUpperBands{TF} = ExactProportionalBands{1,:upper,TF}
const ExactThirdOctaveUpperBands{TF} = ExactProportionalBands{3,:upper,TF}

lower_bands(bands::ExactProportionalBands{NO,LCU,TF}) where {NO,LCU,TF} = ExactProportionalBands{NO,:lower,TF}(bands.bstart, bands.bend)
center_bands(bands::ExactProportionalBands{NO,LCU,TF}) where {NO,LCU,TF} = ExactProportionalBands{NO,:center,TF}(bands.bstart, bands.bend)
upper_bands(bands::ExactProportionalBands{NO,LCU,TF}) where {NO,LCU,TF} = ExactProportionalBands{NO,:upper,TF}(bands.bstart, bands.bend)

const approx_3rd_octave_cbands_pattern = [1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0]
const approx_3rd_octave_lbands_pattern = [0.9, 1.12, 1.4, 1.8, 2.24, 2.8, 3.35, 4.5, 5.6, 7.1]
const approx_3rd_octave_ubands_pattern = [1.12, 1.4, 1.8, 2.24, 2.8, 3.35, 4.5, 5.6, 7.1, 9.0]

struct ApproximateThirdOctaveBands{LCU,TF} <: AbstractProportionalBands{3,LCU,TF}
    bstart::Int
    bend::Int

    function ApproximateThirdOctaveBands{LCU,TF}(bstart::Int, bend::Int) where {LCU, TF}
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU type must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        return new{LCU,TF}(bstart, bend)
    end
    function ApproximateThirdOctaveBands{LCU}(TF, bstart::Int, bend::Int) where {LCU}
        return ApproximateThirdOctaveBands{LCU,TF}(bstart, bend)
    end
end

ApproximateThirdOctaveBands{LCU}(bstart::Int, bend::Int) where {LCU} = ApproximateThirdOctaveBands{LCU,Float64}(bstart, bend)

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:center,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_3rd_octave_cbands_pattern[b]*TF(10)^factor10
end

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:lower,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_3rd_octave_lbands_pattern[b]*TF(10)^factor10
end

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:upper,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_3rd_octave_ubands_pattern[b]*TF(10)^factor10
end

@inline function band_approx_3rd_octave_lower_limit(fl::TF) where {TF}
    factor10 = floor(Int, log10(fl/approx_3rd_octave_lbands_pattern[1]))
    i = searchsortedfirst(approx_3rd_octave_lbands_pattern, fl; lt=(lband, f)->isless(lband*TF(10)^factor10, f))
    # - 2 because
    #
    #   * -1 for searchsortedfirst giving us the first index in approx_3rd_octave_lbands_pattern that is greater than fl, and we want the band before that
    #   * -1 because the array approx_3rd_octave_lbands_pattern is 1-based, but the third-octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 2) + factor10*10
end

@inline function band_approx_3rd_octave_upper_limit(fu::TF) where {TF}
    factor10 = floor(Int, log10(fu/approx_3rd_octave_lbands_pattern[1]))
    i = searchsortedfirst(approx_3rd_octave_ubands_pattern, fu; lt=(uband, f)->isless(uband*TF(10)^factor10, f))
    # - 1 because
    #
    #   * -1 because the array approx_3rd_octave_lbands_pattern is 1-based, but the third-octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 1) + factor10*10
end

# Get the range of approximate third-octave bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ApproximateThirdOctaveBands{LCU}(fstart::TF, fend::TF) where {LCU,TF} = ApproximateThirdOctaveBands{LCU,TF}(fstart, fend)
ApproximateThirdOctaveBands{LCU,TF}(fstart::TF, fend::TF) where {LCU,TF} = ApproximateThirdOctaveBands{LCU,TF}(band_approx_3rd_octave_lower_limit(fstart), band_approx_3rd_octave_upper_limit(fend))

const ApproximateThirdOctaveCenterBands{TF} = ApproximateThirdOctaveBands{:center,TF}
const ApproximateThirdOctaveLowerBands{TF} = ApproximateThirdOctaveBands{:lower,TF}
const ApproximateThirdOctaveUpperBands{TF} = ApproximateThirdOctaveBands{:upper,TF}

const approx_octave_cbands_pattern = [1.0, 2.0, 4.0, 8.0, 16.0, 31.5, 63.0, 125.0, 250.0, 500.0]
const approx_octave_lbands_pattern = [0.71, 1.42, 2.84, 5.68, 11.0, 22.0, 44.0, 88.0, 177.0, 355.0]
const approx_octave_ubands_pattern = [1.42, 2.84, 5.68, 11.0, 22.0, 44.0, 88.0, 177.0, 355.0, 710.0]

struct ApproximateOctaveBands{LCU,TF} <: AbstractProportionalBands{1,LCU,TF}
    bstart::Int
    bend::Int

    function ApproximateOctaveBands{LCU,TF}(bstart::Int, bend::Int) where {LCU, TF}
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU type must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        return new{LCU,TF}(bstart, bend)
    end
    function ApproximateOctaveBands{LCU}(TF, bstart::Int, bend::Int) where {LCU}
        return ApproximateOctaveBands{LCU,TF}(bstart, bend)
    end
end

ApproximateOctaveBands{LCU}(bstart::Int, bend::Int) where {LCU} = ApproximateOctaveBands{LCU,Float64}(bstart, bend)

@inline function Base.getindex(bands::ApproximateOctaveBands{:center,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_octave_cbands_pattern[b]*TF(1000)^factor1000
end

@inline function Base.getindex(bands::ApproximateOctaveBands{:lower,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_octave_lbands_pattern[b]*TF(1000)^factor1000
end

@inline function Base.getindex(bands::ApproximateOctaveBands{:upper,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return approx_octave_ubands_pattern[b]*TF(1000)^factor1000
end

@inline function band_approx_octave_lower_limit(fl::TF) where {TF}
    factor1000 = floor(Int, log10(fl/approx_octave_lbands_pattern[1])/3)
    i = searchsortedfirst(approx_octave_lbands_pattern, fl; lt=(lband, f)->isless(lband*TF(10)^(3*factor1000), f))
    # - 2 because
    #
    #   * -1 for searchsortedfirst giving us the first index in approx_octave_lbands_pattern that is greater than fl, and we want the band before that
    #   * -1 because the array approx_octave_lbands_pattern is 1-based, but the octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 2) + factor1000*10
end

@inline function band_approx_octave_upper_limit(fu::TF) where {TF}
    factor1000 = floor(Int, log10(fu/approx_octave_lbands_pattern[1])/3)
    i = searchsortedfirst(approx_octave_ubands_pattern, fu; lt=(lband, f)->isless(lband*TF(10)^(3*factor1000), f))
    # - 1 because
    #
    #   * -1 because the array approx_octave_lbands_pattern is 1-based, but the octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 1) + factor1000*10
end

# Get the range of approximate octave bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ApproximateOctaveBands{LCU}(fstart::TF, fend::TF) where {LCU,TF} = ApproximateOctaveBands{LCU,TF}(fstart, fend)
ApproximateOctaveBands{LCU,TF}(fstart::TF, fend::TF) where {LCU,TF} = ApproximateOctaveBands{LCU,TF}(band_approx_octave_lower_limit(fstart), band_approx_octave_upper_limit(fend))

const ApproximateOctaveCenterBands{TF} = ApproximateOctaveBands{:center,TF}
const ApproximateOctaveLowerBands{TF} = ApproximateOctaveBands{:lower,TF}
const ApproximateOctaveUpperBands{TF} = ApproximateOctaveBands{:upper,TF}
struct ProportionalBandSpectrum{NO,TF,TAmp,TBandsL<:AbstractProportionalBands{NO,:lower,TF},TBandsC<:AbstractProportionalBands{NO,:center,TF},TBandsU<:AbstractProportionalBands{NO,:upper,TF}} <: AbstractVector{TF}
    f1_nb::TF
    df_nb::TF
    psd_amp::TAmp
    lbands::TBandsL
    cbands::TBandsC
    ubands::TBandsU

    function ProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands{NO}}, f1_nb, df_nb, psd_amp) where {NO}
        TF = promote_type(typeof(f1_nb), typeof(df_nb), eltype(psd_amp))

        f1_nb > zero(f1_nb) || throw(ArgumentError("f1_nb must be > 0"))
        # We're thinking of each non-zero freqeuncy as being a bin with center frequency `f` and width `df_nb`.
        # So to get the lowest non-zero frequency we'll subtract 0.5*df_nb from the lowest non-zero frequency center:
        fstart = max(f1_nb - 0.5*df_nb, TF(fmin_exact))
        fend = f1_nb + (length(psd_amp)-1)*df_nb + 0.5*df_nb

        lbands = TBands{:lower}(fstart, fend)
        cbands = TBands{:center}(TF, band_start(lbands), band_end(lbands))
        ubands = TBands{:upper}(TF, band_start(lbands), band_end(lbands))

        return new{NO,TF,typeof(psd_amp),typeof(lbands), typeof(cbands), typeof(ubands)}(f1_nb, df_nb, psd_amp, lbands, cbands, ubands)
    end
end

const ExactOctaveSpectrum{TF,TAmp} = ProportionalBandSpectrum{1,TF,TAmp,
                                                              ExactProportionalBands{1,:lower,TF},
                                                              ExactProportionalBands{1,:center,TF},
                                                              ExactProportionalBands{1,:upper,TF}}
ExactOctaveSpectrum(f1_nb, df_nb, psd_amp) = ProportionalBandSpectrum(ExactProportionalBands{1}, f1_nb, df_nb, psd_amp)
ExactOctaveSpectrum(sm::AbstractNarrowbandSpectrum) = ProportionalBandSpectrum(ExactProportionalBands{1}, sm)

const ExactThirdOctaveSpectrum{TF,TAmp} = ProportionalBandSpectrum{3,TF,TAmp,
                                                                   ExactProportionalBands{3,:lower,TF},
                                                                   ExactProportionalBands{3,:center,TF},
                                                                   ExactProportionalBands{3,:upper,TF}}
ExactThirdOctaveSpectrum(f1_nb, df_nb, psd_amp) = ProportionalBandSpectrum(ExactProportionalBands{3}, f1_nb, df_nb, psd_amp)
ExactThirdOctaveSpectrum(sm::AbstractNarrowbandSpectrum) = ProportionalBandSpectrum(ExactProportionalBands{3}, sm)

const ApproximateOctaveSpectrum{TF,TAmp} = ProportionalBandSpectrum{1,TF,TAmp,
                                                              ApproximateOctaveBands{:lower,TF},
                                                              ApproximateOctaveBands{:center,TF},
                                                              ApproximateOctaveBands{:upper,TF}}
ApproximateOctaveSpectrum(f1_nb, df_nb, psd_amp) = ProportionalBandSpectrum(ApproximateOctaveBands, f1_nb, df_nb, psd_amp)
ApproximateOctaveSpectrum(sm::AbstractNarrowbandSpectrum) = ProportionalBandSpectrum(ApproximateOctaveBands, sm)

const ApproximateThirdOctaveSpectrum{TF,TAmp} = ProportionalBandSpectrum{1,TF,TAmp,
                                                              ApproximateThirdOctaveBands{:lower,TF},
                                                              ApproximateThirdOctaveBands{:center,TF},
                                                              ApproximateThirdOctaveBands{:upper,TF}}
ApproximateThirdOctaveSpectrum(f1_nb, df_nb, psd_amp) = ProportionalBandSpectrum(ApproximateThirdOctaveBands, f1_nb, df_nb, psd_amp)
ApproximateThirdOctaveSpectrum(sm::AbstractNarrowbandSpectrum) = ProportionalBandSpectrum(ApproximateThirdOctaveBands, sm)

frequency_nb(pbs::ProportionalBandSpectrum) = pbs.f1_nb .+ (0:length(pbs.psd_amp)-1).*pbs.df_nb

function ProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands}, sm::AbstractNarrowbandSpectrum)
    psd = PowerSpectralDensityAmplitude(sm)
    freq = frequency(psd)
    f1_nb = freq[begin+1]
    df_nb = step(freq)
    # Skip the zero frequency.
    psd_amp = @view psd[begin+1:end]
    return ProportionalBandSpectrum(TBands, f1_nb, df_nb, psd_amp)
end

@inline lower_bands(pbs::ProportionalBandSpectrum) = pbs.lbands
@inline center_bands(pbs::ProportionalBandSpectrum) = pbs.cbands
@inline upper_bands(pbs::ProportionalBandSpectrum) = pbs.ubands

@inline Base.size(pbs::ProportionalBandSpectrum) = size(center_bands(pbs))

# Don't need this, right?
# @inline Base.eltype(::Type{ProportionalBandSpectrum{NO,TF}}) where {NO,TF}= TF

@inline function Base.getindex(pbs::ProportionalBandSpectrum, i::Int)
    @boundscheck checkbounds(pbs, i)
    # This is where the fun begins.
    # So, first I want the lower and upper bands of this band.
    fl = lower_bands(pbs)[i]
    fu = upper_bands(pbs)[i]
    # Now I need to find the starting and ending indices that are included in this frequency band.

    # Need the narrowband frequencies.
    # This will not include the zero frequency.
    f_nb = frequency_nb(pbs)

    # This is the narrowband frequency spacing.
    Δf = pbs.df_nb

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
    # First, get all of the psd amplitudes.
    psd_amp = pbs.psd_amp
    # Now get the amplitudes we actually want.
    psd_amp_v = @view psd_amp[istart:iend]
    f_nb_v = @view f_nb[istart:iend]

    # Get the contribution of the first band, which might not be a full band.
    # So the band will start at fl, the lower edge of the proportional band, and
    # end at the narrowband center frequency + 0.5*Δf.
    # This isn't right if the "narrowband" is actually wider than the
    # proportional bands. If that's the case, then we need to clip it to the proportional band width.
    band_lhs = max(f_nb_v[1] - 0.5*Δf, fl)
    band_rhs = min(f_nb_v[1] + 0.5*Δf, fu)
    res_first_band = psd_amp_v[1]*(band_rhs - band_lhs)
    # @show i res_first_band

    # Get the contribution of the last band, which might not be a full band.
    if length(psd_amp_v) > 1
        band_lhs = max(f_nb_v[end] - 0.5*Δf, fl)
        band_rhs = min(f_nb_v[end] + 0.5*Δf, fu)
        res_last_band = psd_amp_v[end]*(band_rhs - band_lhs)
    else
        res_last_band = zero(eltype(pbs))
    end

    # Get all the others and return them.
    psd_amp_v2 = @view psd_amp_v[2:end-1]
    res = res_first_band + res_last_band
    return res_first_band + sum(psd_amp_v2*Δf) + res_last_band
end
