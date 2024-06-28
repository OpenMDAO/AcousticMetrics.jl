"""
    AbstractProportionalBands{NO,LCU,TF} <: AbstractVector{TF}

Abstract type representing the exact proportional frequency bands with band fraction `NO` and `eltype` `TF`.

The `LCU` parameter can take one of three values:

* `:lower`: The `struct` returns the lower edges of each frequency band.
* `:center`: The `struct` returns the center of each frequency band.
* `:upper`: The `struct` returns the upper edges of each frequency band.
"""
abstract type AbstractProportionalBands{NO,LCU,TF} <: AbstractVector{TF} end

"""
    octave_fraction(bands::AbstractProportionalBands{NO}) where {NO}

Return `NO`, the "octave fraction," e.g. `1` for octave bands, `3` for third-octave, `12` for twelfth-octave.
"""
octave_fraction(::AbstractProportionalBands{NO}) where {NO} = NO
octave_fraction(::Type{<:AbstractProportionalBands{NO}}) where {NO} = NO

"""
    lower_center_upper(bands::AbstractProportionalBands{NO,LCU,TF}) where {NO,LCU,TF}

Return `LCU`, which can be either `:lower`, `:center`, `:upper`, indicating if `bands` represents the lower edges, centers, or upper edges of proportional bands, respectively.
"""
lower_center_upper(bands::AbstractProportionalBands{NO,LCU,TF}) where {NO,LCU,TF} = lower_center_upper(typeof(bands))
lower_center_upper(::Type{<:AbstractProportionalBands{NO,LCU,TF}}) where {NO,LCU,TF} = LCU

"""
    freq_scaler(bands::AbstractProportionalBands)

Return the factor each "standard" frequency band is scaled by.

For example, the approximate octave center bands include 1000 Hz, 2000 Hz, and 4000 Hz.
If `freq_scaler(bands) == 1.0`, then these frequencies would be unchanged.
If `freq_scaler(bands) == 1.5`, then `bands` would include 1500 Hz, 3000 Hz, and 6000 Hz instead.
If `freq_scaler(bands) == 0.5`, then `bands` would include 500 Hz, 1000 Hz, and 2000 Hz in place of 1000 Hz, 2000 Hz, and 4000 Hz.
"""
@inline freq_scaler(bands::AbstractProportionalBands) = bands.scaler

"""
    band_start(bands::AbstractProportionalBands)

Return the standard band index number for the first band in `bands`.

For example, it happens that the approximate octave center bands includes 1000 Hz, and that particular band is numbered `10`.
So if the first band contained in `bands` happens to be 1000 Hz (and `freq_scaler(bands) == 1.0`), then `band_start(bands) == 10`.
Not particularly useful to a user.
"""
@inline band_start(bands::AbstractProportionalBands) = bands.bstart

"""
    band_end(bands::AbstractProportionalBands)

Return the standard band index number for the last band in `bands`.

For example, it happens that the approximate octave center bands includes 1000 Hz, and that particular band is numbered `10`.
So if the last band contained in `bands` happens to be 1000 Hz (and `freq_scaler(bands) == 1.0`), then `band_end(bands) == 10`.
Not particularly useful to a user.
"""
@inline band_end(bands::AbstractProportionalBands) = bands.bend

@inline function Base.size(bands::AbstractProportionalBands)
    return (band_end(bands) - band_start(bands) + 1,)
end

"""
   lower_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF} 

Construct and return the lower edges of the proportional bands `TBands`, scaled by `scaler`, that would fully encompass a frequency range beginning with `fstart` and ending with `fend`.
"""
function lower_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF}
    return TBands{:lower}(fstart, fend, scaler)
end

"""
   upper_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF} 

Construct and return the upper edges of the proportional bands `TBands`, scaled by `scaler`, that would fully encompass a frequency range beginning with `fstart` and ending with `fend`.
"""
function upper_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF}
    return TBands{:upper}(fstart, fend, scaler)
end

"""
   center_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF} 

Construct and return the centers of the proportional bands `TBands`, scaled by `scaler`, that would fully encompass a frequency range beginning with `fstart` and ending with `fend`.
"""
function center_bands(TBands::Type{<:AbstractProportionalBands{NO}}, fstart::TF, fend::TF, scaler=1) where {NO,TF}
    return TBands{:center}(fstart, fend, scaler)
end

"""
    cband_number(bands::AbstractProportionalBands, fc)

Return the standard band index number of the band with center frequency `fc` for proportional bands `bands`.

For example, if `bands` is a subtype of `ApproximateOctaveBands` and `freq_scaler(bands) == 1.0`, then `cband_number(bands, 1000.0) == 10`.
"""
cband_number(bands::AbstractProportionalBands, fc) = cband_number(typeof(bands), fc, freq_scaler(bands))

const f0_exact = 1000
const fmin_exact = 1

"""
    ExactProportionalBands{NO,LCU,TF} <: AbstractProportionalBands{NO,LCU,TF}

Representation of the exact proportional frequency bands with band fraction `NO` and `eltype` `TF`.

The `LCU` parameter can take one of three values:

* `:lower`: The `struct` returns the lower edges of each frequency band.
* `:center`: The `struct` returns the center of each frequency band.
* `:upper`: The `struct` returns the upper edges of each frequency band.
"""
ExactProportionalBands

struct ExactProportionalBands{NO,LCU,TF} <: AbstractProportionalBands{NO,LCU,TF}
    bstart::Int
    bend::Int
    f0::TF
    scaler::TF
    function ExactProportionalBands{NO,LCU,TF}(bstart::Int, bend::Int, scaler=1) where {NO,LCU,TF}
        NO > 0 || throw(ArgumentError("Octave band fraction NO must be greater than 0"))
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        scaler > 0 || throw(ArgumentError("non-positive scaler argument not supported"))
        return new{NO,LCU,TF}(bstart, bend, TF(f0_exact), TF(scaler))
    end
end

"""
    ExactProportionalBands{NO,LCU}(TF=Float64, bstart::Int, bend::Int, scaler=1)

Construct an `ExactProportionalBands` with `eltype` `TF` encomposing band index numbers from `bstart` to `bend`.

The "standard" band frequencies will be scaled by `scaler`, e.g. if `scaler = 0.5` then what would normally be the `1000 Hz` frequency will be `500 Hz`, etc..
"""
function ExactProportionalBands{NO,LCU}(TF::Type, bstart::Int, bend::Int, scalar=1) where {NO,LCU}
    return ExactProportionalBands{NO,LCU,TF}(bstart, bend, scalar)
end
function ExactProportionalBands{NO,LCU}(bstart::Int, bend::Int, scaler=1) where {NO,LCU} 
    return ExactProportionalBands{NO,LCU}(Float64, bstart, bend, scaler)
end

@inline band_exact_lower_limit(NO, fl, scaler) = floor(Int, 1/2 + NO*log2(fl/(f0_exact*scaler)) + 10*NO)
@inline band_exact_upper_limit(NO, fu, scaler) = ceil(Int, -1/2 + NO*log2(fu/(f0_exact*scaler)) + 10*NO)

function _cband_exact(NO, fc, scaler)
    # f = 2^((b - 10*NO)/NO)*f0
    # f/f0 = 2^((b - 10*NO)/NO)
    # log2(f/f0) = log2(2^((b - 10*NO)/NO))
    # log2(f/f0) = ((b - 10*NO)/NO)
    # log2(f/f0)*NO = b - 10*NO
    # log2(f/f0)*NO + 10*NO = b
    # b = log2(f/f0)*NO + 10*NO

    # Get the band number from a center band frequency `fc`.
    log2_fc_over_f0_exact_NO = log2(fc/(f0_exact*scaler))*NO

    # Check that the result will be very close to an integer.
    rounded = round(Int, log2_fc_over_f0_exact_NO)
    tol = 10*eps(fc)
    abs_cs_safe(log2_fc_over_f0_exact_NO - rounded) < tol || throw(ArgumentError("fc does not correspond to a center-band frequency"))

    b = rounded + 10*NO
    return b
end

function cband_number(::Type{<:ExactProportionalBands{NO}}, fc, scaler) where {NO}
    return _cband_exact(NO, fc, scaler)
end

"""
    ExactProportionalBands{NO,LCU}(fstart::TF, fend::TF, scaler)

Construct an `ExactProportionalBands` with `eltype` `TF`, scaled by `scaler`, encomposing the bands needed to completely extend over minimum frequency `fstart` and maximum frequency `fend`.
"""
ExactProportionalBands{NO,LCU}(fstart::TF, fend::TF, scaler=1) where {NO,LCU,TF} = ExactProportionalBands{NO,LCU,TF}(fstart, fend, scaler)
ExactProportionalBands{NO,LCU,TF}(fstart::TF, fend::TF, scaler=1) where {NO,LCU,TF} = ExactProportionalBands{NO,LCU,TF}(band_exact_lower_limit(NO, fstart, scaler), band_exact_upper_limit(NO, fend, scaler), scaler)

"""
    Base.getindex(bands::ExactProportionalBands{NO,LCU}, i::Int) where {NO,LCU}

Return the lower, center, or upper frequency (depending on the value of `LCU`) associated with the `i`-th proportional band frequency covered by `bands`.
"""
Base.getindex(bands::ExactProportionalBands, i::Int)

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
    return 2^((b - 10*NO)/NO)*(bands.f0*freq_scaler(bands))
end

@inline function Base.getindex(bands::ExactProportionalBands{NO,:lower}, i::Int) where {NO}
    @boundscheck checkbounds(bands, i)
    b = bands.bstart + (i - 1)
    # return 2^((b - 10*NO)/NO)*(2^(-1/(2*NO)))*bands.f0
    # return 2^(2*(b - 10*NO)/(2*NO))*(2^(-1/(2*NO)))*bands.f0
    return 2^((2*(b - 10*NO) - 1)/(2*NO))*(bands.f0*freq_scaler(bands))
end

@inline function Base.getindex(bands::ExactProportionalBands{NO,:upper}, i::Int) where {NO}
    @boundscheck checkbounds(bands, i)
    b = bands.bstart + (i - 1)
    # return 2^((b - 10*NO)/NO)*(2^(1/(2*NO)))*bands.f0
    # return 2^(2*(b - 10*NO)/(2*NO))*(2^(1/(2*NO)))*bands.f0
    return 2^((2*(b - 10*NO) + 1)/(2*NO))*(bands.f0*freq_scaler(bands))
end

"""
    ExactOctaveCenterBands{TF}

Alias for `ExactProportionalBands{1,:center,TF}`
"""
const ExactOctaveCenterBands{TF} = ExactProportionalBands{1,:center,TF}

"""
    ExactThirdOctaveCenterBands{TF}

Alias for `ExactProportionalBands{3,:center,TF}`
"""
const ExactThirdOctaveCenterBands{TF} = ExactProportionalBands{3,:center,TF}

"""
    ExactOctaveLowerBands{TF}

Alias for `ExactProportionalBands{1,:lower,TF}`
"""
const ExactOctaveLowerBands{TF} = ExactProportionalBands{1,:lower,TF}

"""
    ExactThirdOctaveLowerBands{TF}

Alias for `ExactProportionalBands{3,:lower,TF}`
"""
const ExactThirdOctaveLowerBands{TF} = ExactProportionalBands{3,:lower,TF}

"""
    ExactOctaveUpperBands{TF}

Alias for `ExactProportionalBands{1,:upper,TF}`
"""
const ExactOctaveUpperBands{TF} = ExactProportionalBands{1,:upper,TF}

"""
    ExactThirdOctaveUpperBands{TF}

Alias for `ExactProportionalBands{3,:upper,TF}`
"""
const ExactThirdOctaveUpperBands{TF} = ExactProportionalBands{3,:upper,TF}

"""
   lower_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,TF} 

Construct and return the lower edges of the proportional bands `bands` scaled by `scaler`.
"""
lower_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,LCU,TF} = ExactProportionalBands{NO,:lower,TF}(band_start(bands), band_end(bands), scaler)

"""
   center_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,TF} 

Construct and return the centers of the proportional bands `bands` scaled by `scaler`.
"""
center_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,LCU,TF} = ExactProportionalBands{NO,:center,TF}(band_start(bands), band_end(bands), scaler)

"""
   upper_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,TF} 

Construct and return the upper edges of the proportional bands `bands` scaled by `scaler`.
"""
upper_bands(bands::ExactProportionalBands{NO,LCU,TF}, scaler=freq_scaler(bands)) where {NO,LCU,TF} = ExactProportionalBands{NO,:upper,TF}(band_start(bands), band_end(bands), scaler)

const approx_3rd_octave_cbands_pattern = [1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0]
const approx_3rd_octave_lbands_pattern = [0.9, 1.12, 1.4, 1.8, 2.24, 2.8, 3.35, 4.5, 5.6, 7.1]
const approx_3rd_octave_ubands_pattern = [1.12, 1.4, 1.8, 2.24, 2.8, 3.35, 4.5, 5.6, 7.1, 9.0]

"""
    ApproximateThirdOctaveBands{LCU,TF} <: AbstractProportionalBands{3,LCU,TF}

Representation of the approximate third-octave proportional frequency bands with `eltype` `TF`.

The `LCU` parameter can take one of three values:

* `:lower`: The `struct` returns the lower edges of each frequency band.
* `:center`: The `struct` returns the center of each frequency band.
* `:upper`: The `struct` returns the upper edges of each frequency band.
"""
struct ApproximateThirdOctaveBands{LCU,TF} <: AbstractProportionalBands{3,LCU,TF}
    bstart::Int
    bend::Int
    scaler::TF

    function ApproximateThirdOctaveBands{LCU,TF}(bstart::Int, bend::Int, scaler=1) where {LCU, TF}
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        scaler > 0 || throw(ArgumentError("non-positive scaler argument not supported"))
        return new{LCU,TF}(bstart, bend, TF(scaler))
    end
end

"""
    ApproximateThirdOctaveBands{LCU}(TF=Float64, bstart::Int, bend::Int, scaler=1)

Construct an `ApproximateThirdOctaveBands` with `eltype` `TF` encomposing band index numbers from `bstart` to `bend`.

The "standard" band frequencies will be scaled by `scaler`, e.g. if `scaler = 0.5` then what would normally be the `1000 Hz` frequency will be `500 Hz`, etc..
"""
function ApproximateThirdOctaveBands{LCU}(TF::Type, bstart::Int, bend::Int, scaler=1) where {LCU}
    return ApproximateThirdOctaveBands{LCU,TF}(bstart, bend, scaler)
end
function ApproximateThirdOctaveBands{LCU}(bstart::Int, bend::Int, scaler=1) where {LCU} 
    return ApproximateThirdOctaveBands{LCU}(Float64, bstart, bend, scaler)
end

"""
    Base.getindex(bands::ApproximateThirdOctaveBands{LCU}, i::Int) where {LCU}

Return the lower, center, or upper frequency (depending on the value of `LCU`) associated with the `i`-th proportional band frequency covered by `bands`.
"""
Base.getindex(bands::ApproximateThirdOctaveBands, i::Int)

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:center,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_3rd_octave_cbands_pattern[b]*TF(10)^factor10
end

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:lower,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_3rd_octave_lbands_pattern[b]*TF(10)^factor10
end

@inline function Base.getindex(bands::ApproximateThirdOctaveBands{:upper,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor10, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_3rd_octave_ubands_pattern[b]*TF(10)^factor10
end

@inline function band_approx_3rd_octave_lower_limit(fl::TF, scaler) where {TF}
    # For the `scaler`, I've been thinking about always leaving the input frequency (here `fl`) alone and modifying the standard bands (here `approx_3rd_octave_lbands_pattern`).
    # But then that would involve multiplying all of `approx_3rd_octave_lbands_pattern`...
    # Or maybe not.
    factor10 = floor(Int, log10(fl/(scaler*approx_3rd_octave_lbands_pattern[1])))
    i = searchsortedfirst(approx_3rd_octave_lbands_pattern, fl; lt=(lband, f)->isless(scaler*lband*TF(10)^factor10, f))
    # - 2 because
    #
    #   * -1 for searchsortedfirst giving us the first index in approx_3rd_octave_lbands_pattern that is greater than fl, and we want the band before that
    #   * -1 because the array approx_3rd_octave_lbands_pattern is 1-based, but the third-octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 2) + factor10*10
end

@inline function band_approx_3rd_octave_upper_limit(fu::TF, scaler) where {TF}
    factor10 = floor(Int, log10(fu/(scaler*approx_3rd_octave_lbands_pattern[1])))
    i = searchsortedfirst(approx_3rd_octave_ubands_pattern, fu; lt=(uband, f)->isless(scaler*uband*TF(10)^factor10, f))
    # - 1 because
    #
    #   * -1 because the array approx_3rd_octave_lbands_pattern is 1-based, but the third-octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 1) + factor10*10
end

function cband_approx_3rd_octave(fc, scaler)
    fc_scaled = fc/scaler
    frac, factor10 = modf(log10(fc_scaled))
    # if (frac < -eps(frac))
    #     frac += 1
    #     factor10 -= 1
    # end
    adj = ifelse(frac < -eps(frac), 1, 0)
    frac += adj
    factor10 -= adj
    cband_pattern_entry = 10^frac
    tol_shift = 0.001
    b = searchsortedfirst(approx_3rd_octave_cbands_pattern, cband_pattern_entry-tol_shift)
    tol_compare = 100*eps(approx_3rd_octave_cbands_pattern[b])
    abs_cs_safe(approx_3rd_octave_cbands_pattern[b] - cband_pattern_entry) < tol_compare || throw(ArgumentError("frequency fc does not correspond to an approximate 3rd-octave center band"))
    b0 = b - 1
    j = 10*Int(factor10) + b0
    return j
end

function cband_number(::Type{<:ApproximateThirdOctaveBands}, fc, scaler)
    return cband_approx_3rd_octave(fc, scaler)
end

"""
    ApproximateThirdOctaveBands{LCU}(fstart::TF, fend::TF, scaler=1)

Construct an `ApproximateThirdOctaveBands` with `eltype` `TF`, scaled by `scaler`, encomposing the bands needed to completely extend over minimum frequency `fstart` and maximum frequency `fend`.
"""
ApproximateThirdOctaveBands{LCU}(fstart::TF, fend::TF, scaler=1) where {LCU,TF} = ApproximateThirdOctaveBands{LCU,TF}(fstart, fend, scaler)
ApproximateThirdOctaveBands{LCU,TF}(fstart::TF, fend::TF, scaler=1) where {LCU,TF} = ApproximateThirdOctaveBands{LCU,TF}(band_approx_3rd_octave_lower_limit(fstart, scaler), band_approx_3rd_octave_upper_limit(fend, scaler), scaler)

"""
    ApproximateThirdOctaveCenterBands{TF}

Alias for `ApproximateThirdOctaveBands{:center,TF}`
"""
const ApproximateThirdOctaveCenterBands{TF} = ApproximateThirdOctaveBands{:center,TF}

"""
    ApproximateThirdOctaveLowerBands{TF}

Alias for `ApproximateThirdOctaveBands{:lower,TF}`
"""
const ApproximateThirdOctaveLowerBands{TF} = ApproximateThirdOctaveBands{:lower,TF}

"""
    ApproximateThirdOctaveUpperBands{TF}

Alias for `ApproximateThirdOctaveBands{:upper,TF}`
"""
const ApproximateThirdOctaveUpperBands{TF} = ApproximateThirdOctaveBands{:upper,TF}

"""
   lower_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF}

Construct and return the lower edges of the proportional bands `bands` scaled by `scaler`.
"""
lower_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateThirdOctaveBands{:lower,TF}(band_start(bands), band_end(bands), scaler)

"""
   center_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF}

Construct and return the centers of the proportional bands `bands` scaled by `scaler`.
"""
center_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateThirdOctaveBands{:center,TF}(band_start(bands), band_end(bands), scaler)

"""
   upper_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF}

Construct and return the upper edges of the proportional bands `bands` scaled by `scaler`.
"""
upper_bands(bands::ApproximateThirdOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateThirdOctaveBands{:upper,TF}(band_start(bands), band_end(bands), scaler)

const approx_octave_cbands_pattern = [1.0, 2.0, 4.0, 8.0, 16.0, 31.5, 63.0, 125.0, 250.0, 500.0]
const approx_octave_lbands_pattern = [0.71, 1.42, 2.84, 5.68, 11.0, 22.0, 44.0, 88.0, 177.0, 355.0]
const approx_octave_ubands_pattern = [1.42, 2.84, 5.68, 11.0, 22.0, 44.0, 88.0, 177.0, 355.0, 710.0]

"""
    ApproximateOctaveBands{LCU,TF} <: AbstractProportionalBands{1,LCU,TF}

Representation of the approximate octave proportional frequency bands with `eltype` `TF`.

The `LCU` parameter can take one of three values:

* `:lower`: The `struct` returns the lower edges of each frequency band.
* `:center`: The `struct` returns the center of each frequency band.
* `:upper`: The `struct` returns the upper edges of each frequency band.
"""
struct ApproximateOctaveBands{LCU,TF} <: AbstractProportionalBands{1,LCU,TF}
    bstart::Int
    bend::Int
    scaler::TF

    function ApproximateOctaveBands{LCU,TF}(bstart::Int, bend::Int, scaler=1) where {LCU, TF}
        LCU in (:lower, :center, :upper) || throw(ArgumentError("LCU must be one of :lower, :center, :upper"))
        bend >= bstart || throw(ArgumentError("bend should be greater than or equal to bstart"))
        scaler > 0 || throw(ArgumentError("non-positive scaler argument not supported"))
        return new{LCU,TF}(bstart, bend, TF(scaler))
    end
end

"""
    ApproximateOctaveBands{LCU,TF}(bstart::Int, bend::Int)

Construct an `ApproximateOctaveBands` with `eltype` `TF` encomposing band index numbers from `bstart` to `bend`.

The "standard" band frequencies will be scaled by `scaler`, e.g. if `scaler = 0.5` then what would normally be the `1000 Hz` frequency will be `500 Hz`, etc..
"""
function ApproximateOctaveBands{LCU}(TF::Type, bstart::Int, bend::Int, scaler=1) where {LCU}
    return ApproximateOctaveBands{LCU,TF}(bstart, bend, scaler)
end
function ApproximateOctaveBands{LCU}(bstart::Int, bend::Int, scaler=1) where {LCU} 
    return ApproximateOctaveBands{LCU}(Float64, bstart, bend, scaler)
end

"""
    Base.getindex(bands::ApproximateOctaveBands{LCU}, i::Int) where {LCU}

Return the lower, center, or upper frequency (depending on the value of `LCU`) associated with the `i`-th proportional band frequency covered by `bands`.
"""
Base.getindex(bands::ApproximateOctaveBands, i::Int)

@inline function Base.getindex(bands::ApproximateOctaveBands{:center,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_octave_cbands_pattern[b]*TF(1000)^factor1000
end

@inline function Base.getindex(bands::ApproximateOctaveBands{:lower,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_octave_lbands_pattern[b]*TF(1000)^factor1000
end

@inline function Base.getindex(bands::ApproximateOctaveBands{:upper,TF}, i::Int) where {TF}
    @boundscheck checkbounds(bands, i)
    j = bands.bstart + i - 1
    factor1000, b0 = divrem(j, 10, RoundDown)
    b = b0 + 1
    return freq_scaler(bands)*approx_octave_ubands_pattern[b]*TF(1000)^factor1000
end

@inline function band_approx_octave_lower_limit(fl::TF, scaler) where {TF}
    factor1000 = floor(Int, log10(fl/(scaler*approx_octave_lbands_pattern[1]))/3)
    i = searchsortedfirst(approx_octave_lbands_pattern, fl; lt=(lband, f)->isless(scaler*lband*TF(10)^(3*factor1000), f))
    # - 2 because
    #
    #   * -1 for searchsortedfirst giving us the first index in approx_octave_lbands_pattern that is greater than fl, and we want the band before that
    #   * -1 because the array approx_octave_lbands_pattern is 1-based, but the octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 2) + factor1000*10
end

@inline function band_approx_octave_upper_limit(fu::TF, scaler) where {TF}
    factor1000 = floor(Int, log10(fu/(scaler*approx_octave_lbands_pattern[1]))/3)
    i = searchsortedfirst(approx_octave_ubands_pattern, fu; lt=(lband, f)->isless(scaler*lband*TF(10)^(3*factor1000), f))
    # - 1 because
    #
    #   * -1 because the array approx_octave_lbands_pattern is 1-based, but the octave band pattern band numbers are 0-based (centerband 1.0 Hz is band number 0, etc..)
    return (i - 1) + factor1000*10
end

function cband_approx_octave(fc, scaler)
    fc_scaled = fc/scaler
    frac, factor1000 = modf(log10(fc_scaled)/log10(1000))
    # if (frac < -eps(frac))
    #     frac += 1
    #     factor1000 -= 1
    # end
    adj = ifelse(frac < -eps(frac), 1, 0)
    frac += adj
    factor1000 -= adj
    cband_pattern_entry = 1000^frac
    tol_shift = 0.001
    b = searchsortedfirst(approx_octave_cbands_pattern, cband_pattern_entry-tol_shift)
    tol_compare = 100*eps(approx_octave_cbands_pattern[b])
    abs_cs_safe(approx_octave_cbands_pattern[b] - cband_pattern_entry) < tol_compare || throw(ArgumentError("frequency f does not correspond to an approximate octave center band"))
    b0 = b - 1
    j = 10*Int(factor1000) + b0
    return j
end

function cband_number(::Type{<:ApproximateOctaveBands}, fc, scaler)
    return cband_approx_octave(fc, scaler)
end

"""
    ApproximateOctaveBands{LCU}(fstart::TF, fend::TF, scaler=1)

Construct an `ApproximateOctaveBands` with `eltype` `TF`, scaled by `scaler`, encomposing the bands needed to completely extend over minimum frequency `fstart` and maximum frequency `fend`.
"""
ApproximateOctaveBands{LCU}(fstart::TF, fend::TF, scaler=1) where {LCU,TF} = ApproximateOctaveBands{LCU,TF}(fstart, fend, scaler)
ApproximateOctaveBands{LCU,TF}(fstart::TF, fend::TF, scaler=1) where {LCU,TF} = ApproximateOctaveBands{LCU,TF}(band_approx_octave_lower_limit(fstart, scaler), band_approx_octave_upper_limit(fend, scaler), scaler)

"""
    ApproximateOctaveCenterBands{TF}

Alias for `ApproximateOctaveBands{:center,TF}`
"""
const ApproximateOctaveCenterBands{TF} = ApproximateOctaveBands{:center,TF}

"""
    ApproximateOctaveLowerBands{TF}

Alias for `ApproximateOctaveBands{:lower,TF}`
"""
const ApproximateOctaveLowerBands{TF} = ApproximateOctaveBands{:lower,TF}

"""
    ApproximateOctaveUpperBands{TF}

Alias for `ApproximateOctaveBands{:upper,TF}`
"""
const ApproximateOctaveUpperBands{TF} = ApproximateOctaveBands{:upper,TF}

"""
    lower_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands))

Construct and return the lower edges of the proportional bands `bands` scaled by `scaler`.
"""
lower_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateOctaveBands{:lower,TF}(band_start(bands), band_end(bands), scaler)

"""
    center_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands))

Construct and return the centers of the proportional bands `bands` scaled by `scaler`.
"""
center_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateOctaveBands{:center,TF}(band_start(bands), band_end(bands), scaler)

"""
    upper_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands))

Construct and return the upper edges of the proportional bands `bands` scaled by `scaler`.
"""
upper_bands(bands::ApproximateOctaveBands{LCU,TF}, scaler=freq_scaler(bands)) where {LCU,TF} = ApproximateOctaveBands{:upper,TF}(band_start(bands), band_end(bands), scaler)

"""
    AbstractProportionalBandSpectrum{NO,TF} <: AbstractVector{TF}

Abstract type representing a proportional band spectrum with band fraction `NO` and `eltype` `TF`.
"""
abstract type AbstractProportionalBandSpectrum{NO,TF} <: AbstractVector{TF} end

"""
    octave_fraction(pbs::AbstractProportionalBandSpectrum{NO}) where {NO}

Return `NO`, the "octave fraction," e.g. `1` for octave bands, `3` for third-octave, `12` for twelfth-octave.
"""
octave_fraction(::AbstractProportionalBandSpectrum{NO}) where {NO} = NO
octave_fraction(::Type{<:AbstractProportionalBandSpectrum{NO}}) where {NO} = NO

"""
    lower_bands(pbs::AbstractProportionalBandSpectrum)

Return the lower edges of the proportional bands associated with the proportional band spectrum `pbs`.
"""
@inline lower_bands(pbs::AbstractProportionalBandSpectrum) = lower_bands(pbs.cbands)

"""
    center_bands(pbs::AbstractProportionalBandSpectrum)

Return the centers of the proportional bands associated with the proportional band spectrum `pbs`.
"""
@inline center_bands(pbs::AbstractProportionalBandSpectrum) = pbs.cbands

"""
    upper_bands(pbs::AbstractProportionalBandSpectrum)

Return the upper edges of the proportional bands associated with the proportional band spectrum `pbs`.
"""
@inline upper_bands(pbs::AbstractProportionalBandSpectrum) = upper_bands(pbs.cbands)


"""
    freq_scaler(pbs::AbstractProportionalBandSpectrum)

Return the factor each "standard" frequency band associated with the proportional band spectrum `pbs` is scaled by.

For example, the approximate octave center bands include 1000 Hz, 2000 Hz, and 4000 Hz.
If `freq_scaler(pbs) == 1.0`, then these frequencies would be unchanged.
If `freq_scaler(pbs) == 1.5`, then `bands` would include 1500 Hz, 3000 Hz, and 6000 Hz instead.
If `freq_scaler(pbs) == 0.5`, then `bands` would include 500 Hz, 1000 Hz, and 2000 Hz in place of 1000 Hz, 2000 Hz, and 4000 Hz.
"""
@inline freq_scaler(pbs::AbstractProportionalBandSpectrum) = freq_scaler(center_bands(pbs))

"""
    has_observer_time(pbs::AbstractProportionalBandSpectrum)

Return `true` if the proportional band spectrum is defined to exist over a limited time, `false` otherwise.
"""
@inline has_observer_time(pbs::AbstractProportionalBandSpectrum) = false

"""
    observer_time(pbs::AbstractProportionalBandSpectrum)

Return the observer time at which the proportional band spectrum is defined to exist.
"""
@inline observer_time(pbs::AbstractProportionalBandSpectrum{NO,TF}) where {NO,TF} = zero(TF)

"""
    timestep(pbs::AbstractProportionalBandSpectrum)

Return the time range over which the proportional band spectrum is defined to exist.
"""
@inline timestep(pbs::AbstractProportionalBandSpectrum{NO,TF}) where {NO,TF} = Inf*one(TF)

"""
    amplitude(pbs::AbstractProportionalBandSpectrum)

Return the underlying `Vector` containing the proportional band spectrum amplitudes contained in `pbs`.
"""
@inline amplitude(pbs::AbstractProportionalBandSpectrum) = pbs.pbs

"""
    time_period(pbs::AbstractArray{<:AbstractProportionalBandSpectrum})

Find the period of time over which the collection of proportional band spectrum `pbs` exists.
"""
function time_period(pbs::AbstractArray{<:AbstractProportionalBandSpectrum})
    tmin, tmax = extrema(observer_time, Iterators.filter(has_observer_time, pbs); init=(Inf, -Inf))
    return tmax - tmin
end

"""
    time_scaler(pbs::AbstractProportionalBandSpectrum{NO,TF}, period)

Find the scaling factor appropriate to multiply the proportional band spectrum `pbs` by that accounts for the duration of time the spectrum exists.

This is used when combining multiple proportional band spectra with the [combine](@ref) function.
"""
time_scaler(pbs::AbstractProportionalBandSpectrum{NO,TF}, period) where {NO,TF} = one(TF)

@inline Base.size(pbs::AbstractProportionalBandSpectrum) = size(center_bands(pbs))

"""
    Base.getindex(pbs::AbstractProportionalBandSpectrum, i::Int)

Return the proportional band spectrum amplitude for the `i`th non-zero band in `pbs`.
"""
@inline function Base.getindex(pbs::AbstractProportionalBandSpectrum, i::Int)
    @boundscheck checkbounds(pbs, i)
    return @inbounds amplitude(pbs)[i]
end

"""
    LazyNBProportionalBandSpectrum{NO,IsTonal,TF,TAmp,TBandsC}

Lazy representation of a proportional band spectrum with octave fraction `NO` and `eltype` `TF` constructed from a narrowband (`NB`) spectrum.

`IsTonal` indicates how the acoustic energy is distributed through the narrow frequency bands:

  * `IsTonal == false` means the acoustic energy is assumed to be evenly distributed thoughout each band
  * `IsTonal == true` means the acoustic energy is assumed to be concentrated at each band center
"""
struct LazyNBProportionalBandSpectrum{NO,IsTonal,TF,TAmp<:AbstractVector{TF},TBandsC<:AbstractProportionalBands{NO,:center}} <: AbstractProportionalBandSpectrum{NO,TF}
    f1_nb::TF
    df_nb::TF
    msp_amp::TAmp
    cbands::TBandsC

    function LazyNBProportionalBandSpectrum{NO,IsTonal,TF,TAmp}(f1_nb::TF, df_nb::TF, msp_amp::TAmp, cbands::AbstractProportionalBands{NO,:center}) where {NO,IsTonal,TF,TAmp<:AbstractVector{TF}}
        f1_nb > zero(f1_nb) || throw(ArgumentError("f1_nb must be > 0"))
        df_nb > zero(df_nb) || throw(ArgumentError("df_nb must be > 0"))
        return new{NO,IsTonal,TF,TAmp,typeof(cbands)}(f1_nb, df_nb, msp_amp, cbands)
    end
end

"""
    LazyNBProportionalBandSpectrum{NO,IsTonal}(f1_nb, df_nb, msp_amp, cbands::AbstractProportionalBands{NO,:center})

Construct a lazy representation of a proportional band spectrum with proportional center bands `cbands` from a narrowband spectrum.

The narrowband frequencies are defined by the first narrowband frequency `f1_nb` and the narrowband frequency spacing `df_nb`.
`msp_amp` is the spectrum of narrowband mean squared pressure amplitude.

`IsTonal` indicates how the acoustic energy is distributed through the narrow frequency bands:

  * `IsTonal == false` means the acoustic energy is assumed to be evenly distributed thoughout each band
  * `IsTonal == true` means the acoustic energy is assumed to be concentrated at each band center
"""
function LazyNBProportionalBandSpectrum{NO,IsTonal}(f1_nb, df_nb, msp_amp, cbands::AbstractProportionalBands{NO,:center}) where {NO,IsTonal}
    TF = eltype(msp_amp)
    TAmp = typeof(msp_amp)
    return LazyNBProportionalBandSpectrum{NO,IsTonal,TF,TAmp}(TF(f1_nb), TF(df_nb), msp_amp, cbands)
end

"""
    LazyNBProportionalBandSpectrum(f1_nb, df_nb, msp_amp, cbands::AbstractProportionalBands{NO,:center}, istonal=false)

Construct a lazy representation of a proportional band spectrum with proportional center bands `cbands` from a narrowband spectrum.

The narrowband frequencies are defined by the first narrowband frequency `f1_nb` and the narrowband frequency spacing `df_nb`.
`msp_amp` is the spectrum of narrowband mean squared pressure amplitude.

`istonal` indicates how the acoustic energy is distributed through the narrow frequency bands:

  * `istonal == false` means the acoustic energy is assumed to be evenly distributed thoughout each band
  * `istonal == true` means the acoustic energy is assumed to be concentrated at each band center
"""
function LazyNBProportionalBandSpectrum(f1_nb, df_nb, msp_amp, cbands::AbstractProportionalBands{NO,:center}, istonal::Bool=false) where {NO}
    return LazyNBProportionalBandSpectrum{NO,istonal}(f1_nb, df_nb, msp_amp, cbands)
end

"""
    LazyNBProportionalBandSpectrum{NO,IsTonal}(TBands::Type{<:AbstractProportionalBands{NO}}, f1_nb, df_nb, msp_amp, scaler=1)

Construct a lazy representation of a proportional band spectrum with proportional band type `TBands` from a narrowband spectrum.

The narrowband frequencies are defined by the first narrowband frequency `f1_nb` and the narrowband frequency spacing `df_nb`.
`msp_amp` is the spectrum of narrowband mean squared pressure amplitude.
The proportional band frequencies will be scaled by `scaler`.

`IsTonal` is a `Bool` indicating how the acoustic energy is distributed through the narrow frequency bands:

  * `IsTonal == false` means the acoustic energy is assumed to be evenly distributed thoughout each band
  * `IsTonal == true` means the acoustic energy is assumed to be concentrated at each band center
"""
function LazyNBProportionalBandSpectrum{NO,false}(TBands::Type{<:AbstractProportionalBands{NO}}, f1_nb, df_nb, msp_amp, scaler=1) where {NO}
    TF = eltype(msp_amp)
    TAmp = typeof(msp_amp)
    # We're thinking of each non-zero freqeuncy as being a bin with center frequency `f` and width `df_nb`.
    # So to get the lowest non-zero frequency we'll subtract 0.5*df_nb from the lowest non-zero frequency center:
    fstart = max(f1_nb - 0.5*df_nb, TF(fmin_exact))
    fend = f1_nb + (length(msp_amp)-1)*df_nb + 0.5*df_nb
    cbands = TBands{:center}(fstart, fend, scaler)

    return LazyNBProportionalBandSpectrum{NO,false,TF,TAmp}(TF(f1_nb), TF(df_nb), msp_amp, cbands)
end
function LazyNBProportionalBandSpectrum{NO,true}(TBands::Type{<:AbstractProportionalBands{NO}}, f1_nb, df_nb, msp_amp, scaler=1) where {NO}
    TF = eltype(msp_amp)
    TAmp = typeof(msp_amp)
    # We're thinking of each non-zero freqeuncy as being an infinitely thin "bin" with center frequency `f` and spacing `df_nb`.
    # So to get the lowest non-zero frequency is f1_nb, and the highest is f1_nb + (length(msp_amp)-1)*df_nb.
    fstart = f1_nb
    fend = f1_nb + (length(msp_amp)-1)*df_nb
    cbands = TBands{:center}(fstart, fend, scaler)

    return LazyNBProportionalBandSpectrum{NO,true,TF,TAmp}(TF(f1_nb), TF(df_nb), msp_amp, cbands)
end

"""
    LazyNBProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands}, f1_nb, df_nb, msp_amp, scaler=1, istonal::Bool=false)

Construct a `LazyNBProportionalBandSpectrum` using proportional bands `TBands` and narrowband mean squared pressure amplitude vector `msp_amp` and optional proportional band frequency scaler `scaler`.

`f1_nb` is the first non-zero narrowband frequency, and `df_nb` is the narrowband frequency spacing.
The `istonal` `Bool` argument, if `true`, indicates the narrowband spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each narrow frequency band.
The proportional band frequencies will be scaled by `scaler`.
"""
function LazyNBProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands{NO}}, f1_nb, df_nb, msp_amp, scaler=1, istonal::Bool=false) where {NO}
    return LazyNBProportionalBandSpectrum{NO,istonal}(TBands, f1_nb, df_nb, msp_amp, scaler)
end

"""
    LazyNBProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands}, sm::AbstractNarrowbandSpectrum, scaler=1)

Construct a `LazyNBProportionalBandSpectrum` using a proportional band `TBands` and narrowband spectrum `sm`, and optional frequency scaler `scaler`.
The proportional band frequencies will be scaled by `scaler`.
"""
function LazyNBProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands{NO}}, sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}, scaler=1) where {NO,IsEven,IsTonal}
    msp = MSPSpectrumAmplitude(sm)
    freq = frequency(msp)
    f1_nb = freq[begin+1]
    df_nb = step(freq)
    # Skip the zero frequency.
    msp_amp = @view msp[begin+1:end]
    return LazyNBProportionalBandSpectrum{NO,IsTonal}(TBands, f1_nb, df_nb, msp_amp, scaler)
end

"""
    LazyNBProportionalBandSpectrum(sm::AbstractNarrowbandSpectrum, cbands::AbstractProportionalBands{NO,:center})

Construct a `LazyNBProportionalBandSpectrum` using proportional centerbands `cbands` and narrowband spectrum `sm`.
The proportional band frequencies will be scaled by `scaler`.
"""
function LazyNBProportionalBandSpectrum(sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}, cbands::AbstractProportionalBands{NO,:center}) where {NO,IsEven,IsTonal}
    msp = MSPSpectrumAmplitude(sm)
    TF = eltype(msp)
    freq = frequency(msp)
    f1_nb = TF(freq[begin+1])
    df_nb = TF(step(freq))
    # Skip the zero frequency.
    msp_amp = @view msp[begin+1:end]
    TAmp = typeof(msp_amp)
    return LazyNBProportionalBandSpectrum{NO,IsTonal,TF,TAmp}(f1_nb, df_nb, msp_amp, cbands)
end

"""
    frequency_nb(pbs::LazyNBProportionalBandSpectrum)

Return the narrowband frequencies associated with the underlying narrowband spectrum contained in `pbs`.
"""
frequency_nb(pbs::LazyNBProportionalBandSpectrum) = pbs.f1_nb .+ (0:length(pbs.msp_amp)-1).*pbs.df_nb

"""
    lazy_pbs(pbs, cbands::AbstractProportionalBands{NO,:center})

Construct a lazy proportional band spectrum on proportional center bands `cbands` using the proportional band spectrum `pbs`.
"""
lazy_pbs

function lazy_pbs(pbs::LazyNBProportionalBandSpectrum{NOIn,IsTonal}, cbands::AbstractProportionalBands{NO,:center}) where {NOIn,IsTonal,NO}
    return LazyNBProportionalBandSpectrum{NO,IsTonal}(pbs.f1_nb, pbs.df_nb, pbs.msp_amp, cbands)
end

"""
    Base.getindex(pbs::LazyNBProportionalBandSpectrum{NO,false}, i::Int)

Return the proportional band spectrum amplitude for the `i`th non-zero band in `pbs` from a non-tonal narrowband.
"""
@inline function Base.getindex(pbs::LazyNBProportionalBandSpectrum{NO,false}, i::Int) where {NO}
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
        return zero(eltype(pbs))
    end

    # Need the msp amplitude relavent for this band.
    # First, get all of the msp amplitudes.
    msp_amp = pbs.msp_amp
    # Now get the amplitudes we actually want.
    msp_amp_v = @view msp_amp[istart:iend]
    f_nb_v = @view f_nb[istart:iend]

    # Get the contribution of the first band, which might not be a full band.
    # So the band will start at fl, the lower edge of the proportional band, and
    # end at the narrowband center frequency + 0.5*Δf.
    # This isn't right if the "narrowband" is actually wider than the
    # proportional bands. If that's the case, then we need to clip it to the proportional band width.
    band_lhs = max(f_nb_v[1] - 0.5*Δf, fl)
    band_rhs = min(f_nb_v[1] + 0.5*Δf, fu)
    res_first_band = msp_amp_v[1]/Δf*(band_rhs - band_lhs)

    # Get the contribution of the last band, which might not be a full band.
    if length(msp_amp_v) > 1
        band_lhs = max(f_nb_v[end] - 0.5*Δf, fl)
        band_rhs = min(f_nb_v[end] + 0.5*Δf, fu)
        res_last_band = msp_amp_v[end]/Δf*(band_rhs - band_lhs)
    else
        res_last_band = zero(eltype(pbs))
    end

    # Get all the others and return them.
    msp_amp_v2 = @view msp_amp_v[2:end-1]
    return res_first_band + sum(msp_amp_v2) + res_last_band
end

"""
    Base.getindex(pbs::LazyNBProportionalBandSpectrum{NO,true}, i::Int)

Return the proportional band spectrum amplitude for the `i`th non-zero band in `pbs` from a tonal narrowband.
"""
@inline function Base.getindex(pbs::LazyNBProportionalBandSpectrum{NO,true}, i::Int) where {NO}
    @boundscheck checkbounds(pbs, i)
    # This is where the fun begins.
    # So, first I want the lower and upper bands of this band.
    fl = lower_bands(pbs)[i]
    # Arg, numerical problems: lower_bands[i+1] should be the same as upper_bands[i].
    # But because of floating point inaccuracies, they can be a tiny bit different.
    # And that can lead to a "gap" between, say, upper_bands[i] and lower_bands[i+1].
    # And then if a tone is right in that gap, we'll miss part of the spectrum.
    # So, to fix this, always use the lower band values except for the last proportional band (where it won't matter, since that frequency value is only used once, and hence there can't be any gap).
    if i < length(pbs)
        fu = lower_bands(pbs)[i+1]
    else
        fu = upper_bands(pbs)[i]
    end
    # Now I need to find the starting and ending indices that are included in this frequency band.

    # Need the narrowband frequencies.
    # This will not include the zero frequency.
    f_nb = frequency_nb(pbs)

    # This is the narrowband frequency spacing.
    Δf = pbs.df_nb

    # So, what is the first index we want?
    # It's the one that has f_nb[i] >= fl.
    istart = searchsortedfirst(f_nb, fl)
    # `searchsortedfirst` will return `length(f_nb)+1` it doesn't find anything.
    # What does that mean?
    # That means that all the frequencies in the narrowband spectrum are lower
    # than the band we're looking at. So return 0.
    if istart == length(f_nb) + 1
        return zero(eltype(pbs))
    end

    # What is the last index we want?
    # It's the last one that has f_nb[i] <= fu
    # iend = searchsortedlast(f_nb, fu)
    # But we don't want to double-count frequencies, so we actually want f_nb[i] < fu.
    # Could just do `searchsortedlast(f_nb, fu; lt=<=)`, but this avoids the possibly-slow keyword arguments.
    iend = searchsortedlast(f_nb, fu, ord(<=, identity, nothing, Forward))
    if iend == 0
        # All the frequencies are lower than the band we're looking for.
        return zero(eltype(pbs))
    end

    # Need the msp amplitude relavent for this band.
    # First, get all of the msp amplitudes.
    msp_amp = pbs.msp_amp
    # Now get the amplitudes we actually want.
    msp_amp_v = @view msp_amp[istart:iend]

    # Since we're thinking of the narrowband frequency bins as being infinitely thin, they can't partially extend beyond the lower or upper limits of the relevant proportional band.
    # So we just need to add them up here:
    return sum(msp_amp_v)
end

"""
    ProportionalBandSpectrum{NO,TF,TPBS,TBandsL,TBandsC,TBandsU}

Representation of a proportional band spectrum with octave fraction `NO` and `eltype` `TF`.
"""
struct ProportionalBandSpectrum{NO,TF,TPBS<:AbstractVector{TF},TBandsC<:AbstractProportionalBands{NO,:center}} <: AbstractProportionalBandSpectrum{NO,TF}
    pbs::TPBS
    cbands::TBandsC

    function ProportionalBandSpectrum(pbs, cbands::AbstractProportionalBands{NO,:center}) where {NO}
        length(pbs) == length(cbands) || throw(ArgumentError("length(pbs) must match length(cbands)"))
        return new{NO,eltype(pbs),typeof(pbs),typeof(cbands)}(pbs, cbands)
    end
end

function lazy_pbs(pbs::ProportionalBandSpectrum, cbands::AbstractProportionalBands{NO,:center}) where {NO}
    return LazyPBSProportionalBandSpectrum(pbs, cbands)
end

"""
    ProportionalBandSpectrum(TBandsC, cfreq_start, pbs, scaler=1)

Construct a `ProportionalBandSpectrum` from an array of proportional band amplitudes and proportional band type `TBandsC`.

`cfreq_start` is the centerband frequency corresponding to the first entry of `pbs`. 
The proportional band frequencies indicated by `TBandsC` are multiplied by `scaler`.
"""
function ProportionalBandSpectrum(TBandsC::Type{<:AbstractProportionalBands{NO,:center}}, cfreq_start, pbs, scaler=1) where {NO}
    bstart = cband_number(TBandsC, cfreq_start, scaler)
    bend = bstart + length(pbs) - 1
    cbands = TBandsC(bstart, bend, scaler)

    return ProportionalBandSpectrum(pbs, cbands)
end

"""
    ProportionalBandSpectrumWithTime{NO,TF,TPBS,TBandsC,TTime,TDTime}

Representation of a proportional band spectrum with octave fraction `NO` and `eltype` `TF`, but with an observer time.
"""
struct ProportionalBandSpectrumWithTime{NO,TF,TPBS<:AbstractVector{TF},TBandsC<:AbstractProportionalBands{NO,:center},TDTime,TTime} <: AbstractProportionalBandSpectrum{NO,TF}
    pbs::TPBS
    cbands::TBandsC
    dt::TDTime
    t::TTime

    @doc """
        ProportionalBandSpectrumWithTime(pbs, cbands::AbstractProportionalBands{NO,:center}, dt, t)

    Construct a proportional band spectrum from mean-squared pressure amplitudes `pbs` and centerband frequencies `cbands`, defined to exist over time range `dt` and at observer time `t`.
    """
    function ProportionalBandSpectrumWithTime(pbs, cbands::AbstractProportionalBands{NO,:center}, dt, t) where {NO}
        length(pbs) == length(cbands) || throw(ArgumentError("length(pbs) must match length(cbands)"))
        dt > zero(dt) || throw(ArgumentError("dt must be positive"))

        return new{NO,eltype(pbs),typeof(pbs),typeof(cbands),typeof(dt),typeof(t)}(pbs, cbands, dt, t)
    end
end

@inline has_observer_time(pbs::ProportionalBandSpectrumWithTime) = true
@inline observer_time(pbs::ProportionalBandSpectrumWithTime) = pbs.t
@inline timestep(pbs::ProportionalBandSpectrumWithTime{NO,TF}) where {NO,TF} = pbs.dt
@inline time_scaler(pbs::ProportionalBandSpectrumWithTime, period) = timestep(pbs)/period

function lazy_pbs(pbs::ProportionalBandSpectrumWithTime, cbands::AbstractProportionalBands{NO,:center}) where {NO}
    return LazyPBSProportionalBandSpectrum(pbs, cbands)
end

"""
    LazyPBSProportionalBandSpectrum{NO,TF} <: AbstractProportionalBandSpectrum{NO,TF}

Lazy representation of a proportional band spectrum with octave fraction `NO` and `eltype` `TF` constructed from a different proportional band spectrum.
"""
struct LazyPBSProportionalBandSpectrum{NO,TF,TPBS<:AbstractProportionalBandSpectrum,TBandsC<:AbstractProportionalBands{NO,:center}} <: AbstractProportionalBandSpectrum{NO,TF}
    pbs::TPBS
    cbands::TBandsC

    function LazyPBSProportionalBandSpectrum(pbs::AbstractProportionalBandSpectrum{NOIn,TF}, cbands::AbstractProportionalBands{NO,:center}) where {NO,TF,NOIn}
        return new{NO,TF,typeof(pbs),typeof(cbands)}(pbs, cbands)
    end
end

function LazyPBSProportionalBandSpectrum(TBands::Type{<:AbstractProportionalBands{NO}}, pbs::AbstractProportionalBandSpectrum, scaler=1) where {NO}
    # First, get the minimum and maximum frequencies associated with the input pbs.
    fstart = lower_bands(pbs)[begin]
    fend = upper_bands(pbs)[end]
    # Now use those frequencies to construct some centerbands.
    cbands = TBands{:center}(fstart, fend, scaler)
    # Now we can create the object.
    return LazyPBSProportionalBandSpectrum(pbs, cbands)
end

@inline has_observer_time(pbs::LazyPBSProportionalBandSpectrum) = has_observer_time(pbs.pbs)
@inline observer_time(pbs::LazyPBSProportionalBandSpectrum) = observer_time(pbs.pbs)
@inline timestep(pbs::LazyPBSProportionalBandSpectrum) = timestep(pbs.pbs)
@inline time_scaler(pbs::LazyPBSProportionalBandSpectrum, period) = time_scaler(pbs.pbs, period)

function lazy_pbs(pbs::LazyPBSProportionalBandSpectrum, cbands::AbstractProportionalBands{NO,:center}) where {NO}
    return LazyPBSProportionalBandSpectrum(pbs.pbs, cbands)
end

@inline function Base.getindex(pbs::LazyPBSProportionalBandSpectrum, i::Int)
    @boundscheck checkbounds(pbs, i)

    # So, first I want the lower and upper bands of this output band.
    fol = lower_bands(pbs)[i]
    fou = upper_bands(pbs)[i]

    # Get the underlying pbs.
    pbs_in = pbs.pbs

    # Get the lower and upper edges of the input band's spectrum.
    inbands_lower = lower_bands(pbs_in)
    inbands_upper = upper_bands(pbs_in)

    # So now I have the boundaries of the frequencies I'm interested in in `fol` and `fou`.
    # What I'm looking for now is:
    #
    #   * the first input band whose upper edge is greater than `fol`
    #   * the last input band whose lower edge is less than `fou`.
    #
    # So, for the first input band whose upper edge is greater than `fol`, I should be able to do this:
    istart = searchsortedfirst(inbands_upper, fol)

    # For that, what if
    #
    #   * All of `inbands_upper` are less than `fol`?
    #     That would mean all of the `inband` frequencies are lower than and outside the current `outband`.
    #     Then the docs for `searchsortedfirst` say that it will return `length(inbands_upper)+1`.
    #     So if I started a view of the data from that index, it would obviously be empty, which is what I'd want.
    #   * All of the `inbands_upper` are greater than `fol`?
    #     Not necessarily a problem, unless, I guess, the lowest of `inbands_lower` is *also* greater than `fou`.
    #     Then the entire input spectrum would be larger than this band.
    #     But `searchsortedfirst` should just return `1`, and hopefully that would be the right thing.

    # Now I want the last input band whose lower edge is less than `fou`.
    # I should be able to get that from
    iend = searchsortedlast(inbands_lower, fou)
    # For that, what if 
    #
    #   * All of the `inbands_lower` are greater than `fou`?
    #     That would mean all of the `inband` frequencies are greater than and outside the current `outband`.
    #     The docs indicate `searchsortedlast` would return `firstindex(inbands_lower)-1` for that case, i.e. `0`.
    #     That's what I'd want, I think.
    #   * All of the `inbands_lower` are lower than `fou`?
    #     Not necessarily a problem, unless the highest of `inbands_upper` are also lower than `fou`, which would mean the entire input spectrum is lower than this output band.


    # Now I have the first and last input bands relevant to this output band, and so I can start adding up the input PBS's contributions to this output band.
    pbs_out = zero(eltype(pbs))

    # First, we need to check that there's something to do:
    if (istart <= lastindex(pbs_in)) && (iend >= firstindex(pbs_in))

        # First, get the bandwidth of the first input band associated with this output band.
        fil_start = inbands_lower[istart]
        fiu_start = inbands_upper[istart]
        dfin_start = fiu_start - fil_start

        # Next, need to get the frequency overlap of the first input band and this output band.
        # For the lower edge of the overlap, it will usually be `fol`, unless there's a gap where `inbands_lower[istart]` is greater than `fol`.
        foverlapl_start = max(fol, fil_start)
        # For the upper edge of the overlap, it will usually be `fiu_start`, unless there's a gap where `inbands_upper[istart]` is less than `fou`.
        foverlapu_start = min(fou, fiu_start)

        # Now get the first band's contribution to the PBS.
        pbs_out += pbs_in[istart]/dfin_start*(foverlapu_start - foverlapl_start)

        # Now, think about the last band's contribution to the PBS.
        # First, we need to check if the first and last band are identicial, which would indicate that there's only one input band in this output band.
        if iend > istart
            # Now need to get the bandwidth associated with this input band.
            fil_end = inbands_lower[iend]
            fiu_end = inbands_upper[iend]
            dfin_end = fiu_end - fil_end

            # Next, need to get the frequency overlap of the last input band and this output band.
            foverlapl_end = max(fol, fil_end)
            foverlapu_end = min(fou, fiu_end)

            # Now we can get the last band's contribution to the PBS.
            pbs_out += pbs_in[iend]/dfin_end*(foverlapu_end - foverlapl_end)

            # Now we need the contribution of the input bands between `istart+1` and `iend-1`, inclusive.
            # Don't need to worry about incomplete overlap of the bands since these are "inside" this output band, so we can just directly sum them.
            pbs_in_v = @view pbs_in[istart+1:iend-1]
            pbs_out += sum(pbs_in_v)
        end
    end

    return pbs_out
end

"""
    combine(pbs::AbstractArray{<:AbstractProportionalBandSpectrum}, outcbands::AbstractProportionalBands{NO,:center}, time_axis=1) where {NO}

Combine each input proportional band spectrum of `pbs` into one output proportional band spectrum using the proportional center bands indicated by `outcbands`.

`time_axis` is an integer indicating the axis of the `pbs` array along which time varies.
For example, if `time_axis == 1` and `pbs` is a three-dimensional array, then `apth[:, i, j]` would be proportional band spectrum of source `i`, `j`  for all time.
But if `time_axis == 3`, then `pbs[i, j, :]` would be the proportional band spectrum of source `i`, `j` for all time.
"""
function combine(pbs::AbstractArray{<:AbstractProportionalBandSpectrum}, outcbands::AbstractProportionalBands{NO,:center}, time_axis=1) where {NO}
    # Create the vector that will contain the new PBS.
    # An <:AbstractProportionalBandSpectrum is <:AbstractVector{TF}, so AbstractArray{<:AbstractProportionalBandSpectrum,N} is actually an Array of AbstractVectors.
    # So `eltype(eltype(pbs))` should give me the element type of the PBS.
    TFOut = promote_type(eltype(eltype(pbs)), eltype(outcbands))
    pbs_out = zeros(TFOut, length(outcbands))

    dims_in = axes(pbs)
    ndims_in = ndims(pbs)
    alldims_in = 1:ndims_in

    otherdims = setdiff(alldims_in, time_axis)
    itershape = tuple(dims_in[otherdims]...)

    # Create an array we'll use to index pbs_in, with a `Colon()` for the time_axis position and integers of the first value for all the others.
    idx = [ifelse(d==time_axis, Colon(), first(ind)) for (d, ind) in enumerate(axes(pbs))]

    nidx = length(otherdims)
    indices = CartesianIndices(itershape)

    # Loop through the indices.
    for I in indices
        for i in 1:nidx
            idx[otherdims[i]] = I.I[i]
        end

        # Grab all the elements associated with this time.
        pbs_v = @view pbs[idx...]

        # Now add this element's contribution to pbs_out.
        _combine!(pbs_out, pbs_v, outcbands)
    end

    return ProportionalBandSpectrum(pbs_out, outcbands)
end

"""
    _combine!(pbs_out::AbstractVector, pbs::AbstractVector{<:AbstractProportionalBandSpectrum}, outcbands::AbstractProportionalBands{NO,:center}) where {N}

Combine each input proportional band spectrum of `pbs` into one output Vector using the proportional center bands indicated by `outcbands`.
"""
function _combine!(pbs_out::AbstractVector, pbs::AbstractVector{<:AbstractProportionalBandSpectrum}, outcbands::AbstractProportionalBands{NO,:center}) where {NO}

    # Get the time period for this collection of PBSs.
    period = time_period(pbs)

    # Now start looping over each input PBS.
    for pbs_in in pbs

        # Get the time scaler associated with this particular input PBS.
        scaler = time_scaler(pbs_in, period)

        # Create a lazy version of the input proportional band spectrum using the output center bands.
        pbs_in_lazy = lazy_pbs(pbs_in, outcbands)

        # Now add this element's contribution to output pbs.
        pbs_out .+= pbs_in_lazy .* scaler
    end

    return nothing
end
