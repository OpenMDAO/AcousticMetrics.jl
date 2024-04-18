"""
    AbstractPressureTimeHistory{IsEven}

Supertype for a pressure time history, i.e., pressure as a function of time defined on evenly-spaced time samples.

The `IsEven` parameter is a `Bool` indicating if the length of the pressure time history is even or not.
"""
abstract type AbstractPressureTimeHistory{IsEven} end

"""
    PressureTimeHistory{IsEven} <: AbstractPressureTimeHistory{IsEven}

Pressure as a function of time defined on evenly-spaced time samples.

The `IsEven` parameter is a `Bool` indicating if the length of the pressure time history is even or not.
"""
struct PressureTimeHistory{IsEven,Tp,Tdt,Tt0} <: AbstractPressureTimeHistory{IsEven}
    p::Tp
    dt::Tdt
    t0::Tt0

    function PressureTimeHistory{IsEven}(p, dt, t0) where {IsEven}
        n = length(p)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(p) = $n"))
        return new{IsEven, typeof(p), typeof(dt), typeof(t0)}(p, dt, t0)
    end
end

"""
    PressureTimeHistory(p, dt, t0=zero(dt))

Construct a `PressureTimeHistory` from a vector of pressures `p`, time spacing `dt`, and initial time `t0`.
"""
function PressureTimeHistory(p, dt, t0=zero(dt))
    # TODO: it would be nice to have a constructor that allows for a default value of t0 and explicitly set the value of the `IsEven` parameter.
    n = length(p)
    return PressureTimeHistory{iseven(n)}(p, dt, t0)
end

"""
    pressure(pth::AbstractPressureTimeHistory)

Return a vector of pressures associated with a pressure time history.
"""
@inline pressure(pth::AbstractPressureTimeHistory) = pth.p

"""
    inputlength(pth::AbstractPressureTimeHistory)

Return a number of pressure samples associated with a pressure time history.
"""
@inline inputlength(pth::AbstractPressureTimeHistory) = length(pressure(pth))

"""
    timestep(pth::AbstractPressureTimeHistory)

Return the time step size `dt` associated with a pressure time history.
"""
@inline timestep(pth::AbstractPressureTimeHistory) = pth.dt

"""
    starttime(pth::AbstractPressureTimeHistory)

Return the initial time `t0` associated with a pressure time history.
"""
@inline starttime(pth::AbstractPressureTimeHistory) = pth.t0

"""
    time(pth::AbstractPressureTimeHistory)

Return a vector of times associated with a pressure time history.
"""
@inline function time(pth::AbstractPressureTimeHistory)
    n = inputlength(pth)
    return starttime(pth) .+ (0:n-1) .* timestep(pth)
end

"""
    AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel} <: AbstractVector{Tel}

Supertype for a generic narrowband acoustic metric which will behave as an immutable `AbstractVector` of element type `Tel`.

The `IsEven` parameter is a `Bool` indicating if the length of the spectrum is even or not, affecting how the Nyquist frequency is calculated.
`IsTonal` indicates how the acoustic energy is distributed through the frequency bands:

  * `IsTonal == false` means the acoustic energy is assumed to be enenly distributed thoughout each band
  * `IsTonal == true` means the acoustic energy is assumed to be concentrated at each band center
"""
abstract type AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel} <: AbstractVector{Tel} end

"""
    halfcomplex(sm::AbstractNarrowbandSpectrum)

Return a vector of the discrete Fourier transform of the pressure time history in half-complex format.

See the FFTW docs for the definition of the [halfcomplex format](https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html).
"""
@inline halfcomplex(sm::AbstractNarrowbandSpectrum) = sm.hc

"""
    timestep(sm::AbstractNarrowbandSpectrum)

Return the time step size `dt` associated with a narrowband spectrum.
"""
@inline timestep(sm::AbstractNarrowbandSpectrum) = sm.dt

"""
    starttime(sm::AbstractNarrowbandSpectrum)

Return the initial time `t0` associated with a pressure time history.
"""
@inline starttime(sm::AbstractNarrowbandSpectrum) = sm.t0

"""
    inputlength(sm::AbstractNarrowbandSpectrum)

Return a number of pressure time samples associated with a narrowband spectrum.

This is also the length of the discrete Fourier transform associated with the spectrum in [half-complex format](https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html).
"""
@inline inputlength(sm::AbstractNarrowbandSpectrum) = length(halfcomplex(sm))

"""
    samplerate(sm::AbstractNarrowbandSpectrum)

Return the sample rate (aka the inverse of the time step size) associated with a narrowband spectrum.
"""
@inline samplerate(sm::AbstractNarrowbandSpectrum) = 1/timestep(sm)

"""
    frequency(sm::AbstractNarrowbandSpectrum)

Return a vector of frequencies associated with the narrowband spectrum.

The frequencies are calculated using the `rfftfreq` function in the FFTW.jl package.
"""
@inline frequency(sm::AbstractNarrowbandSpectrum) = rfftfreq(inputlength(sm), samplerate(sm))


"""
    frequencystep(sm::AbstractNarrowbandSpectrum)

Return the frequency step size `Δf` associated with the narrowband spectrum.
"""
@inline function frequencystep(sm::AbstractNarrowbandSpectrum)
    m = inputlength(sm)
    df = 1/(timestep(sm)*m)
    return df
end

"""
    istonal(sm::AbstractNarrowbandSpectrum)

Return `true` if the spectrum is tonal, `false` otherwise.
"""
@inline istonal(sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}) where {IsEven,IsTonal} = IsTonal

"""
    PressureTimeHistory(sm::AbstractNarrowbandSpectrum, p=similar(halfcomplex(sm)))

Construct a pressure time history from a narrowband spectrum `sm`.

The optional `p` argument will be used to store the pressure vector of the pressure time history, and should have length `inputlength(sm)`.
"""
function PressureTimeHistory(sm::AbstractNarrowbandSpectrum, p=similar(halfcomplex(sm)))
    hc = halfcomplex(sm)

    # Get the inverse FFT of the pressure spectrum.
    irfft!(p, hc)

    # Need to divide by the input length since FFTW computes an "unnormalized" FFT.
    p ./= inputlength(sm)

    return PressureTimeHistory(p, timestep(sm), starttime(sm))
end

@inline function Base.size(sm::AbstractNarrowbandSpectrum)
    # So, what's the maximum and minimum index?
    # Minimum is 1, aka 0 + 1.
    # Max is n/2 (rounded down) + 1
    n = inputlength(sm)
    return (n>>1 + 1,)
end

"""
    PressureSpectrumAmplitude{IsEven,IsTonal,Tel} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}

Representation of acoustic pressure amplitude as a function of narrowband frequency.

The `IsEven` parameter is a `Bool` indicating if the length of the spectrum is even or not, affecting how the Nyquist frequency is calculated.
The `IsTonal` `Bool` parameter, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
struct PressureSpectrumAmplitude{IsEven,IsTonal,Tel,Thc,Tdt,Tt0} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PressureSpectrumAmplitude{IsEven,IsTonal}(hc, dt, t0) where {IsEven,IsTonal}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        typeof(IsTonal) === Bool || throw(ArgumentError("typeof(IsTonal) should be Bool"))
        return new{IsEven, IsTonal, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

"""
    PressureSpectrumAmplitude(hc, dt, t0=zero(dt), istonal=false)

Construct a narrowband spectrum of the pressure amplitude from the discrete Fourier transform in half-complex format `hc`, time step size `dt`, and initial time `t0`.
The `istonal` `Bool` argument, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
function PressureSpectrumAmplitude(hc, dt, t0=zero(dt), istonal=false)
    n = length(hc)
    return PressureSpectrumAmplitude{iseven(n),istonal}(hc, dt, t0)
end

"""
    PressureSpectrumAmplitude(sm::AbstractNarrowbandSpectrum)

Construct a narrowband spectrum of the pressure amplitude from another narrowband spectrum.
"""
PressureSpectrumAmplitude(sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}) where {IsEven,IsTonal} = PressureSpectrumAmplitude{IsEven,IsTonal}(halfcomplex(sm), timestep(sm), starttime(sm))

"""
    PressureSpectrumAmplitude(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))

Construct a narrowband spectrum of the pressure amplitude from a pressure time history.

The optional argument `hc` will be used to store the discrete Fourier transform of the pressure time history, and should have length of `inputlength(pth)`.
The `istonal` `Bool` argument, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
function PressureSpectrumAmplitude(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PressureSpectrumAmplitude(hc, timestep(pth), starttime(pth), istonal)
end

@inline function Base.getindex(psa::PressureSpectrumAmplitude{false}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    if i == 1
        @inbounds hc_real = psa.hc[i]/m
        return abs(hc_real)
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*sqrt(hc_real^2 + hc_imag^2)
    end
end

@inline function Base.getindex(psa::PressureSpectrumAmplitude{true}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    if i == 1 || i == length(psa)
        @inbounds hc_real = psa.hc[i]/m
        return abs(hc_real)
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*sqrt(hc_real^2 + hc_imag^2)
    end
end

"""
    PressureSpectrumPhase{IsEven,IsTonal,Tel} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}

Representation of acoustic pressure phase as a function of narrowband frequency.

The `IsEven` parameter is a `Bool` indicating if the length of the spectrum is even or not, affecting how the Nyquist frequency is calculated.
The `IsTonal` `Bool` parameter, if `true`, indicates the phase spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
struct PressureSpectrumPhase{IsEven,IsTonal,Tel,Thc,Tdt,Tt0} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PressureSpectrumPhase{IsEven,IsTonal}(hc, dt, t0) where {IsEven,IsTonal}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        typeof(IsTonal) === Bool || throw(ArgumentError("typeof(IsTonal) should be Bool"))
        return new{IsEven, IsTonal, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

"""
    PressureSpectrumPhase(hc, dt, t0=zero(dt), istonal=false)

Construct a narrowband spectrum of the pressure phase from the discrete Fourier transform in half-complex format `hc`, time step size `dt`, and initial time `t0`.
"""
function PressureSpectrumPhase(hc, dt, t0=zero(dt), istonal=false)
    n = length(hc)
    return PressureSpectrumPhase{iseven(n),istonal}(hc, dt, t0)
end

"""
    PressureSpectrumPhase(sm::AbstractNarrowbandSpectrum)

Construct a narrowband spectrum of the pressure phase from another narrowband spectrum.
"""
PressureSpectrumPhase(sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}) where {IsEven,IsTonal} = PressureSpectrumPhase{IsEven,IsTonal}(halfcomplex(sm), timestep(sm), starttime(sm))

"""
    PressureSpectrumPhase(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))

Construct a narrowband spectrum of the pressure phase from a pressure time history.

The optional argument `hc` will be used to store the discrete Fourier transform of the pressure time history, and should have length of `inputlength(pth)`.
The `istonal` `Bool` argument, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
function PressureSpectrumPhase(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PressureSpectrumPhase(hc, timestep(pth), starttime(pth), istonal)
end

@inline function Base.getindex(psp::PressureSpectrumPhase{false}, i::Int)
    @boundscheck checkbounds(psp, i)
    m = inputlength(psp)
    if i == 1
        @inbounds hc_real = psp.hc[i]/m
        hc_imag = zero(eltype(halfcomplex(psp)))
        phase_t0 = atan(hc_imag, hc_real)
    else
        @inbounds hc_real = psp.hc[i]/m
        @inbounds hc_imag = psp.hc[m-i+2]/m
        phase_t0 = atan(hc_imag, hc_real)
    end
    return rem2pi(phase_t0 - 2*pi*frequency(psp)[i]*starttime(psp), RoundNearest)
end

@inline function Base.getindex(psp::PressureSpectrumPhase{true}, i::Int)
    @boundscheck checkbounds(psp, i)
    m = inputlength(psp)
    if i == 1 || i == length(psp)
        @inbounds hc_real = psp.hc[i]/m
        hc_imag = zero(eltype(halfcomplex(psp)))
        phase_t0 = atan(hc_imag, hc_real)
    else
        @inbounds hc_real = psp.hc[i]/m
        @inbounds hc_imag = psp.hc[m-i+2]/m
        phase_t0 = atan(hc_imag, hc_real)
    end
    return rem2pi(phase_t0 - 2*pi*frequency(psp)[i]*starttime(psp), RoundNearest)
end

"""
    MSPSpectrumAmplitude{IsEven,IsTonal,Tel} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}

Representation of mean-squared pressure amplitude as a function of narrowband frequency.

The `IsEven` parameter is a `Bool` indicating if the length of the spectrum is even or not, affecting how the Nyquist frequency is calculated.
The `IsTonal` `Bool` parameter, if `true`, indicates the mean-squared pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the pressure spectrum is assumed to be constant over each frequency band.
"""
struct MSPSpectrumAmplitude{IsEven,IsTonal,Tel,Thc,Tdt,Tt0} <: AbstractNarrowbandSpectrum{IsEven,IsTonal,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function MSPSpectrumAmplitude{IsEven,IsTonal}(hc, dt, t0) where {IsEven,IsTonal}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        typeof(IsTonal) === Bool || throw(ArgumentError("typeof(IsTonal) should be Bool"))
        return new{IsEven, IsTonal, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

"""
    MSPSpectrumAmplitude(hc, dt, t0=zero(dt), istonal=false)

Construct a narrowband spectrum of the mean-squared pressure amplitude from the discrete Fourier transform in half-complex format `hc`, time step size `dt`, and initial time `t0`.
The `istonal` `Bool` argument, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
function MSPSpectrumAmplitude(hc, dt, t0=zero(dt), istonal=false)
    n = length(hc)
    return MSPSpectrumAmplitude{iseven(n),istonal}(hc, dt, t0)
end

"""
    MSPSpectrumAmplitude(sm::AbstractNarrowbandSpectrum)

Construct a narrowband spectrum of the mean-squared pressure amplitude from another narrowband spectrum.
"""
MSPSpectrumAmplitude(sm::AbstractNarrowbandSpectrum{IsEven,IsTonal}) where {IsEven,IsTonal} = MSPSpectrumAmplitude{IsEven,IsTonal}(halfcomplex(sm), timestep(sm), starttime(sm))

"""
    MSPSpectrumAmplitude(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))

Construct a narrowband spectrum of the mean-squared pressure amplitude from a pressure time history.

The optional argument `hc` will be used to store the discrete Fourier transform of the pressure time history, and should have length of `inputlength(pth)`.
The `istonal` `Bool` argument, if `true`, indicates the pressure spectrum is tonal and thus concentrated at discrete frequencies.
If `false`, the spectrum is assumed to be constant over each frequency band.
"""
function MSPSpectrumAmplitude(pth::AbstractPressureTimeHistory, istonal=false, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return MSPSpectrumAmplitude(hc, timestep(pth), starttime(pth), istonal)
end

@inline function Base.getindex(psa::MSPSpectrumAmplitude{false}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    if i == 1
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)
    end
end

@inline function Base.getindex(psa::MSPSpectrumAmplitude{true}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    if i == 1 || i == length(psa)
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)
    end
end

"""
    MSPSpectrumPhase

Alias for `PressureSpectrumPhase`.
"""
const MSPSpectrumPhase = PressureSpectrumPhase

"""
    PowerSpectralDensityAmplitude{IsEven,Tel} <: AbstractNarrowbandSpectrum{IsEven,false,Tel}

Representation of acoustic power spectral density amplitude as a function of narrowband frequency.

The `IsEven` parameter is a `Bool` indicating if the length of the spectrum is even or not, affecting how the Nyquist frequency is calculated.
As the power spectral density is not well-defined for tones, the `IsTonal` parameter is always `false`.
"""
struct PowerSpectralDensityAmplitude{IsEven,Tel,Thc,Tdt,Tt0} <: AbstractNarrowbandSpectrum{IsEven,false,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PowerSpectralDensityAmplitude{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

"""
    PowerSpectralDensityAmplitude(hc, dt, t0=zero(dt))

Construct a narrowband spectrum of the power spectral density amplitude from the discrete Fourier transform in half-complex format `hc`, time step size `dt`, and initial time `t0`.
"""
function PowerSpectralDensityAmplitude(hc, dt, t0=zero(dt))
    n = length(hc)
    return PowerSpectralDensityAmplitude{iseven(n)}(hc, dt, t0)
end

"""
    PowerSpectralDensityAmplitude(sm::AbstractNarrowbandSpectrum)

Construct a narrowband spectrum of the power spectral density amplitude from another narrowband spectrum.
"""
PowerSpectralDensityAmplitude(sm::AbstractNarrowbandSpectrum{IsEven,false}) where {IsEven} = PowerSpectralDensityAmplitude(halfcomplex(sm), timestep(sm), starttime(sm))
PowerSpectralDensityAmplitude(sm::AbstractNarrowbandSpectrum{IsEven,true}) where {IsEven} = throw(ArgumentError("IsTonal == true parameter cannot be used with PowerSpectralDensityAmplitude type"))

"""
    PowerSpectralDensityAmplitude(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))

Construct a narrowband spectrum of the power spectral density amplitude from a pressure time history.

The optional argument `hc` will be used to store the discrete Fourier transform of the pressure time history, and should have length of `inputlength(pth)`.
"""
function PowerSpectralDensityAmplitude(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PowerSpectralDensityAmplitude(hc, timestep(pth), starttime(pth))
end

@inline function Base.getindex(psa::PowerSpectralDensityAmplitude{false}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    df = frequencystep(psa)
    if i == 1
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2/df
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)/df
    end
end

@inline function Base.getindex(psa::PowerSpectralDensityAmplitude{true}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    df = frequencystep(psa)
    if i == 1 || i == length(psa)
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2/df
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)/df
    end
end

"""
    PowerSpectralDensityPhase

Alias for `PressureSpectrumPhase`.
"""
const PowerSpectralDensityPhase = PressureSpectrumPhase

"""
    OASPL(ap::AbstractPressureTimeHistory)

Return the overall sound pressure level of a pressure time history.
"""
function OASPL(ap::AbstractPressureTimeHistory)
    p = pressure(ap)
    n = inputlength(ap)
    p_mean = sum(p)/n
    msp = sum((p .- p_mean).^2)/n
    return 10*log10(msp/p_ref^2)
end

"""
    OASPL(ap::AbstractNarrowbandSpectrum)

Return the overall sound pressure level of a narrowband spectrum.
"""
function OASPL(sp::AbstractNarrowbandSpectrum)
    amp = MSPSpectrumAmplitude(sp)
    msp = sum(@view amp[begin+1:end])
    return 10*log10(msp/p_ref^2)
end
