abstract type AbstractPressureTimeHistory{IsEven} end

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

# TODO: it would be nice to have a constructor that allows for a default value of t0 and explicitly set the value of the `IsEven` parameter.
function PressureTimeHistory(p, dt, t0=zero(dt))
    n = length(p)
    return PressureTimeHistory{iseven(n)}(p, dt, t0)
end

@inline pressure(pth::AbstractPressureTimeHistory) = pth.p
@inline inputlength(pth::AbstractPressureTimeHistory) = length(pressure(pth))
@inline timestep(pth::AbstractPressureTimeHistory) = pth.dt
@inline starttime(pth::AbstractPressureTimeHistory) = pth.t0
@inline function time(pth::AbstractPressureTimeHistory)
    n = inputlength(pth)
    return starttime(pth) .+ (0:n-1) .* timestep(pth)
end

abstract type AbstractSpectrumMetric{IsEven,Tel} <: AbstractVector{Tel} end

@inline halfcomplex(sm::AbstractSpectrumMetric) = sm.hc
@inline timestep(sm::AbstractSpectrumMetric) = sm.dt
@inline starttime(sm::AbstractSpectrumMetric) = sm.t0
@inline inputlength(sm::AbstractSpectrumMetric) = length(halfcomplex(sm))
@inline samplerate(sm::AbstractSpectrumMetric) = 1/timestep(sm)
@inline frequency(sm::AbstractSpectrumMetric) = rfftfreq(inputlength(sm), samplerate(sm))

function PressureTimeHistory(sm::AbstractSpectrumMetric, p=similar(halfcomplex(sm)))
    hc = halfcomplex(sm)

    # Get the inverse FFT of the pressure spectrum.
    irfft!(p, hc)

    # Need to divide by the input length since FFTW computes an "unnormalized" FFT.
    p ./= inputlength(sm)

    return PressureTimeHistory(p, timestep(sm), starttime(sm))
end

@inline function Base.size(sm::AbstractSpectrumMetric)
    # So, what's the maximum and minimum index?
    # Minimum is 1, aka 0 + 1.
    # Max is n/2 (rounded down) + 1
    n = inputlength(sm)
    return (n>>1 + 1,)
end

struct PressureSpectrumAmplitude{IsEven,Tel,Thc,Tdt,Tt0} <: AbstractSpectrumMetric{IsEven,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PressureSpectrumAmplitude{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function PressureSpectrumAmplitude(hc, dt, t0=zero(dt))
    n = length(hc)
    return PressureSpectrumAmplitude{iseven(n)}(hc, dt, t0)
end

PressureSpectrumAmplitude(sm::AbstractSpectrumMetric) = PressureSpectrumAmplitude(halfcomplex(sm), timestep(sm), starttime(sm))

function PressureSpectrumAmplitude(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PressureSpectrumAmplitude(hc, timestep(pth), starttime(pth))
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

struct PressureSpectrumPhase{IsEven,Tel,Thc,Tdt,Tt0} <: AbstractSpectrumMetric{IsEven,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PressureSpectrumPhase{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function PressureSpectrumPhase(hc, dt, t0=zero(dt))
    n = length(hc)
    return PressureSpectrumPhase{iseven(n)}(hc, dt, t0)
end

PressureSpectrumPhase(sm::AbstractSpectrumMetric) = PressureSpectrumPhase(halfcomplex(sm), timestep(sm), starttime(sm))

function PressureSpectrumPhase(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PressureSpectrumPhase(hc, timestep(pth), starttime(pth))
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

struct NarrowbandSpectrumAmplitude{IsEven,Tel,Thc,Tdt,Tt0} <: AbstractSpectrumMetric{IsEven,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function NarrowbandSpectrumAmplitude{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function NarrowbandSpectrumAmplitude(hc, dt, t0=zero(dt))
    n = length(hc)
    return NarrowbandSpectrumAmplitude{iseven(n)}(hc, dt, t0)
end

NarrowbandSpectrumAmplitude(sm::AbstractSpectrumMetric) = NarrowbandSpectrumAmplitude(halfcomplex(sm), timestep(sm), starttime(sm))

function NarrowbandSpectrumAmplitude(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return NarrowbandSpectrumAmplitude(hc, timestep(pth), starttime(pth))
end

@inline function Base.getindex(psa::NarrowbandSpectrumAmplitude{false}, i::Int)
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

@inline function Base.getindex(psa::NarrowbandSpectrumAmplitude{true}, i::Int)
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

const NarrowbandSpectrumPhase = PressureSpectrumPhase

struct PowerSpectralDensityAmplitude{IsEven,Tel,Thc,Tdt,Tt0} <: AbstractSpectrumMetric{IsEven,Tel}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PowerSpectralDensityAmplitude{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, eltype(hc), typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function PowerSpectralDensityAmplitude(hc, dt, t0=zero(dt))
    n = length(hc)
    return PowerSpectralDensityAmplitude{iseven(n)}(hc, dt, t0)
end

PowerSpectralDensityAmplitude(sm::AbstractSpectrumMetric) = PowerSpectralDensityAmplitude(halfcomplex(sm), timestep(sm), starttime(sm))

function PowerSpectralDensityAmplitude(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PowerSpectralDensityAmplitude(hc, timestep(pth), starttime(pth))
end

@inline function Base.getindex(psa::PowerSpectralDensityAmplitude{false}, i::Int)
    @boundscheck checkbounds(psa, i)
    m = inputlength(psa)
    df = 1/(timestep(psa)*m)
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
    df = 1/(timestep(psa)*m)
    if i == 1 || i == length(psa)
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2/df
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)/df
    end
end

const PowerSpectralDensityPhase = PressureSpectrumPhase

function OASPL(ap::AbstractPressureTimeHistory)
    p = pressure(ap)
    n = inputlength(ap)
    p_mean = sum(p)/n
    msp = sum((p .- p_mean).^2)/n
    return 10*log10(msp/p_ref^2)
end

function OASPL(sp::AbstractSpectrumMetric)
    amp = NarrowbandSpectrumAmplitude(sp)
    msp = sum(@view amp[begin+1:end])
    return 10*log10(msp/p_ref^2)
end
