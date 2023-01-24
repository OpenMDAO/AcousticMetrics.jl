const f0_exact = 1000
const fmin_exact = 1

@inline band_exact_lower(NO, fl) = floor(Int, 1/2 + NO*log2(fl/f0_exact) + 10*NO)
@inline band_exact_upper(NO, fu) = ceil(Int, -1/2 + NO*log2(fu/f0_exact) + 10*NO)

struct ExactProportionalBands{NO,LCU,TF} <: AbstractVector{TF}
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

ExactProportionalBands{NO,LCU}(bstart::Int, bend::Int) where {NO,LCU} = ExactProportionalBands{NO,LCU}(Float64, bstart, bend)

# Get the range of exact center bands necessary to completely extend over a range of frequencies from `fstart` to `fend`.
ExactProportionalBands{NO,LCU}(fstart::TF, fend::TF) where {NO,LCU,TF} = ExactProportionalBands{NO,LCU,TF}(band_exact_lower(NO, fstart), band_exact_upper(NO, fend))


@inline function Base.size(bands::ExactProportionalBands)
    return (bands.bend - bands.bstart + 1,)
end

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

struct ExactProportionalBandSpectrum{NO,TF,TAmp} <: AbstractVector{TF}
    lbands::ExactProportionalBands{NO,:lower,TF}
    cbands::ExactProportionalBands{NO,:center,TF}
    ubands::ExactProportionalBands{NO,:upper,TF}
    f1_nb::TF
    df_nb::TF
    psd_amp::TAmp

    function ExactProportionalBandSpectrum{NO}(f1_nb, df_nb, psd_amp) where {NO}
        f1_nb > zero(f1_nb) || throw(ArgumentError("f1_nb must be > 0"))
        # First frequency is always zero, and the frequencies are always evenly spaced, so the second frequency is the same as the spacing.
        # Δf = psd_freq[begin+1]
        # We're thinking of each non-zero freqeuncy as being a bin with center
        # frequency `f` and width `df_nb`. So to get the lowest non-zero frequency
        # we'll subtract 0.5*df_nb from the lowest non-zero frequency center:
        #   fstart = f[2] - 0.5*df_nb = f[2] - 0.5(f[2]) = 0.5*f[2] = 0.5*df_nb
        fstart = max(f1_nb - 0.5*df_nb, fmin_exact)
        # fend = last(psd_freq) + Δf
        fend = f1_nb + (length(psd_amp)-1)*df_nb + 0.5*df_nb

        lbands = ExactProportionalBands{NO,:lower}(fstart, fend)
        cbands = ExactProportionalBands{NO,:center}(lbands.bstart, lbands.bend)
        ubands = ExactProportionalBands{NO,:upper}(lbands.bstart, lbands.bend)

        TF = promote_type(typeof(f1_nb), typeof(df_nb), eltype(psd_amp))

        return new{NO,TF,typeof(psd_amp)}(lbands, cbands, ubands, f1_nb, df_nb, psd_amp)
    end
end

const ExactOctaveSpectrum{TF,TAmp} = ExactProportionalBandSpectrum{1,TF,TAmp}
const ExactThirdOctaveSpectrum{TF,TAmp} = ExactProportionalBandSpectrum{3,TF,TAmp}

frequency_nb(pbs::ExactProportionalBandSpectrum) = pbs.f1_nb .+ (0:length(pbs.psd_amp)-1).*pbs.df_nb

function ExactProportionalBandSpectrum{NO}(sm::AbstractSpectrumMetric) where {NO}
    psd = PowerSpectralDensityAmplitude(sm)
    freq = frequency(psd)
    f1_nb = freq[begin+1]
    df_nb = step(freq)
    # Skip the zero frequency.
    psd_amp = @view psd[begin+1:end]
    return ExactProportionalBandSpectrum{NO}(f1_nb, df_nb, psd_amp)
end

@inline function Base.size(pbs::ExactProportionalBandSpectrum)
    return (pbs.lbands.bend - pbs.lbands.bstart + 1,)
end

@inline Base.eltype(::Type{ExactProportionalBandSpectrum{NO,TF}}) where {NO,TF}= TF

@inline lower_bands(pbs::ExactProportionalBandSpectrum) = pbs.lbands
@inline center_bands(pbs::ExactProportionalBandSpectrum) = pbs.cbands
@inline upper_bands(pbs::ExactProportionalBandSpectrum) = pbs.ubands

@inline function Base.getindex(pbs::ExactProportionalBandSpectrum, i::Int)
    @boundscheck checkbounds(pbs, i)
    # This is where the fun begins.
    # So, first I want the lower and upper bands of this band.
    fl = lower_bands(pbs)[i]
    fu = upper_bands(pbs)[i]
    # Now I need to find the starting and ending indices that are included in
    # this frequency band.

    # Need the narrowband frequencies, but skip the zero frequency.
    # f_nb = @view pbs.psd_freq[begin+1:end]
    f_nb = frequency_nb(pbs)

    # This is the narrowband frequency spacing.
    # Δf = f_nb[begin]
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
    # Need the `istart+1` and `iend+1` to adjust for dropping the zero-frequency component.
    # psd_amp_v = @view psd_amp[istart+1:iend+1]
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

    # Get the contribution of the last band, which might not be a full band.
    if length(psd_amp_v) > 1
        band_lhs = max(f_nb_v[end] - 0.5*Δf, fl)
        band_rhs = min(f_nb_v[end] + 0.5*Δf, fu)
        res_last_band = psd_amp_v[end]*(band_rhs - band_lhs)
    else
        res_last_band = zero(eltype(pbs))
    end

    # Get all the others and return them.
    return res_first_band + sum(psd_amp_v[2:end-1]*Δf) + res_last_band
end
