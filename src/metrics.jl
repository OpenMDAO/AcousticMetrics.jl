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
    return starttime(ap) .+ (0:n-1) .* timestep(ap)
end

abstract type AbstractSpectrum{IsEven} end

@inline halfcomplex(s::AbstractSpectrum) = s.hc
@inline timestep(s::AbstractSpectrum) = s.dt
@inline starttime(s::AbstractSpectrum) = s.t0
@inline inputlength(s::AbstractSpectrum) = length(halfcomplex(s))
@inline samplerate(s::AbstractSpectrum) = 1/timestep(s)
@inline frequency(s::AbstractSpectrum) = rfftfreq(inputlength(s), samplerate(s))

abstract type AbstractPressureSpectrum{IsEven} <: AbstractSpectrum{IsEven} end

struct PressureSpectrum{IsEven,Thc,Tdt,Tt0} <: AbstractPressureSpectrum{IsEven}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function PressureSpectrum{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function PressureSpectrum(hc, dt, t0=zero(dt))
    n = length(hc)
    return PressureSpectrum{iseven(n)}(hc, dt, t0)
end

function PressureSpectrum(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return PressureSpectrum(hc, timestep(pth), starttime(pth))
end

abstract type AbstractSpectrumMetric{IsEven,Tel} <: AbstractVector{Tel} end

@inline halfcomplex(sm::AbstractSpectrumMetric) = sm.hc
@inline timestep(sm::AbstractSpectrumMetric) = sm.dt
@inline starttime(sm::AbstractSpectrumMetric) = sm.t0
@inline inputlength(sm::AbstractSpectrumMetric) = length(halfcomplex(sm))
@inline samplerate(sm::AbstractSpectrumMetric) = 1/timestep(sm)
@inline frequency(sm::AbstractSpectrumMetric) = rfftfreq(inputlength(sm), samplerate(sm))

@inline function Base.size(psm::AbstractSpectrumMetric)
    # So, what's the maximum and minimum index?
    # Minimum is 1, aka 0 + 1.
    # Max is n/2 (rounded down) + 1
    n = inputlength(psm)
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

@inline function Base.getindex(psa::PressureSpectrumAmplitude{false}, i::Int)
    n = length(psa)
    @boundscheck 1 ≤ i ≤ n
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
    n = length(psa)
    @boundscheck 1 ≤ i ≤ n
    m = inputlength(psa)
    if i == 1 || i == n
        @inbounds hc_real = psa.hc[i]/m
        return abs(hc_real)
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*sqrt(hc_real^2 + hc_imag^2)
    end
end

@inline amplitude(ps::AbstractPressureSpectrum) = PressureSpectrumAmplitude(halfcomplex(ps), timestep(ps), starttime(ps))

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

@inline function Base.getindex(psp::PressureSpectrumPhase{false}, i::Int)
    n = length(psp)
    @boundscheck 1 ≤ i ≤ n
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
    n = length(psp)
    @boundscheck 1 ≤ i ≤ n
    m = inputlength(psp)
    if i == 1 || i == n
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

@inline phase(ps::AbstractPressureSpectrum) = PressureSpectrumPhase(halfcomplex(ps), timestep(ps), starttime(ps))

function PressureTimeHistory(ps::AbstractPressureSpectrum, p=similar(halfcomplex(ps)))
    hc = halfcomplex(ps)

    # Get the inverse FFT of the pressure spectrum.
    irfft!(p, hc)

    # Need to divide by the input length since FFTW computes an "unnormalized" FFT.
    p ./= inputlength(ps)

    return PressureTimeHistory(p, timestep(ps), starttime(ps))
end

abstract type AbstractNarrowbandSpectrum{IsEven} <: AbstractSpectrum{IsEven} end

struct NarrowbandSpectrum{IsEven,Thc,Tdt,Tt0} <: AbstractNarrowbandSpectrum{IsEven}
    hc::Thc
    dt::Tdt
    t0::Tt0

    function NarrowbandSpectrum{IsEven}(hc, dt, t0) where {IsEven}
        n = length(hc)
        iseven(n) == IsEven || throw(ArgumentError("IsEven = $(IsEven) is not consistent with length(hc) = $n"))
        return new{IsEven, typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
    end
end

function NarrowbandSpectrum(hc, dt, t0=zero(dt))
    n = length(hc)
    return NarrowbandSpectrum{iseven(n)}(hc, dt, t0)
end

function NarrowbandSpectrum(pth::AbstractPressureTimeHistory, hc=similar(pressure(pth)))
    p = pressure(pth)

    # Get the FFT of the acoustic pressure.
    rfft!(hc, p)

    return NarrowbandSpectrum(hc, timestep(pth), starttime(pth))
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

@inline function Base.getindex(psa::NarrowbandSpectrumAmplitude{false}, i::Int)
    n = length(psa)
    @boundscheck 1 ≤ i ≤ n
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
    n = length(psa)
    @boundscheck 1 ≤ i ≤ n
    m = inputlength(psa)
    if i == 1 || i == n
        @inbounds hc_real = psa.hc[i]/m
        return hc_real^2
    else
        @inbounds hc_real = psa.hc[i]/m
        @inbounds hc_imag = psa.hc[m-i+2]/m
        return 2*(hc_real^2 + hc_imag^2)
    end
end

@inline amplitude(nbs::AbstractNarrowbandSpectrum) = NarrowbandSpectrumAmplitude(halfcomplex(nbs), timestep(nbs), starttime(nbs))
@inline phase(nbs::AbstractNarrowbandSpectrum) = PressureSpectrumPhase(halfcomplex(nbs), timestep(nbs), starttime(nbs))
@inline PressureSpectrum(nbs::AbstractNarrowbandSpectrum) = PressureSpectrum(halfcomplex(nbs), timestep(nbs), starttime(nbs))
@inline PressureTimeHistory(nbs::AbstractNarrowbandSpectrum) = PressureTimeHistory(PressureSpectrum(nbs))

#@inline inputlength(ps::AbstractPressureSpectrum) = length(halfcomplex(ps))
#@inline timestep(ps::AbstractPressureSpectrum) = ps.dt
#@inline samplerate(ps::AbstractPressureSpectrum) = 1/timestep(ps)
#@inline starttime(ps::AbstractPressureSpectrum) = ps.t0
#@inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
#PressureTimeHistory(p, dt) = AcousticPressure(p, dt, zero(typeof(dt)))

#@inline pressure(ap::AbstractPressureTimeHistory) = ap.p
#@inline inputlength(ap::AbstractPressureTimeHistory) = length(pressure(ap))
#@inline timestep(ap::AbstractPressureTimeHistory) = ap.dt
#@inline starttime(ap::AbstractPressureTimeHistory) = ap.t0

#@inline function time(ap::AbstractPressureTimeHistory)
#    n = length(pressure(ap))
#    return starttime(ap) .+ (0:n-1) .* timestep(ap)
#end

#@inline function size_hc_real(hc)
#    n = length(hc)
#    n_real = n >> 1
#    return n_real
#end

#@inline function size_hc_imag(hc)
#    n = length(hc)
#    n_imag = ((n + 1) >> 1) - 1
#    return n_imag
#end

#@inline function size_amp_phase(hc)
#    n_real = size_hc_real(hc)
#    return 1 + n_real
#end

#"""
#    split_hc_real_imag(hc::AbstractVector)

#Split the output of a Fourier Transform in [halfcomplex
#format](https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html) into
#its mean, real, and imaginary parts.

#Returns a length-three tuple with the mean (a scalar) first, then a `@view` of
#the real components, and a `@view` of the imaginary components.
#"""
#function split_hc_real_imag(hc)
#    n_real = size_hc_real(hc)
#    n_imag = size_hc_imag(hc)

#    # The mean of the signal is in the first element of the output array.
#    # Always real for a real-input FFT.
#    hc_mean = hc[begin]
#    # Then the real parts of the positive frequencies.
#    hc_real = @view hc[begin+1:n_real+1]
#    # Then the imaginary parts of the positive frequencies, which are "backwards."
#    hc_imag = @view hc[end:-1:end-n_imag+1]

#    return hc_mean, hc_real, hc_imag
#end

#abstract type AbstractPressureSpectrum{IsEven} end

#struct PressureSpectrum{IsEven,Thc,Tdt,Tt0} <: AbstractPressureSpectrum{IsEven}
#    hc::Thc
#    dt::Tdt
#    t0::Tt0

#    function PressureSpectrum{IsEven}(hc, dt, t0) where {IsEven}
#        n = length(hc)
#        iseven(n) == IsEven || error("IsEven = $(IsEven) is not consistent with length(hc) = $n")
#        return new{IsEven, typeof(hc), typeof(dt), typeof(t0)}(hc, dt, t0)
#    end
#end

#PressureSpectrum(hc, dt) = PressureSpectrum(hc, dt, zero(typeof(dt)))

## @concrete struct PressureSpectrum <: AbstractPressureSpectrum
##     amp
##     ϕ
##     n
##     fs
##     t0
## end

## @concrete struct PressureSpectrum <: AbstractPressureSpectrum
##     hc
##     fs
##     t0
## end

## PressureSpectrum(hc, fs) = PressureSpectrum(hc, fs, zero(typeof(fs)))

## @inline inputlength(ps::AbstractPressureSpectrum) = ps.n
## @inline samplerate(ps::AbstractPressureSpectrum) = ps.fs
## @inline starttime(ps::AbstractPressureSpectrum) = ps.t0
## @inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
## @inline amplitude(ps::AbstractPressureSpectrum) = ps.amp
## @inline phase(ps::AbstractPressureSpectrum) = rem2pi.(ps.ϕ .- 2 .* pi .* frequency(ps) .* starttime(ps), RoundNearest)
#@inline halfcomplex(ps::AbstractPressureSpectrum) = ps.hc
#@inline inputlength(ps::AbstractPressureSpectrum) = length(halfcomplex(ps))
#@inline timestep(ps::AbstractPressureSpectrum) = ps.dt
#@inline samplerate(ps::AbstractPressureSpectrum) = 1/timestep(ps)
#@inline starttime(ps::AbstractPressureSpectrum) = ps.t0
#@inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
## @inline amplitude(ps::AbstractPressureSpectrum) = ps.amp
#@inline phase(ps::AbstractPressureSpectrum) = rem2pi.(phase_t0(ps) .- 2 .* pi .* frequency(ps) .* starttime(ps), RoundNearest)

#function PressureSpectrum{IsEven}(pth::AbstractPressureTimeHistory{IsEven}, buf=similar(pressure(pth))) where {IsEven}
#    p = pressure(pth)
#    n = inputlength(pth)

#    # # Get the FFT of the acoustic pressure. Divide by n since FFT computes an
#    # # "unnormalized" FFT.
#    # p_fft = rfft(p)./n

#    # # Split the FFT output in half-complex format into mean, real, and imaginary
#    # # components (returns views for the latter two).
#    # p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

#    # # output length.
#    # n_real = length(p_fft_real)
#    # m = 1 + n_real

#    # # Now I want to get the amplitude and phase.
#    # amp = similar(p, m)
#    # phi = similar(p, m)
#    # # Set the amplitude and phase for the mean component.
#    # amp[begin] = abs(p_fft_mean)
#    # phi[begin] = atan(zero(eltype(phi)), p_fft_mean)
#    # if mod(n, 2) == 0
#    #     # There is one more real component than imaginary component (not
#    #     # counting the mean).
#    #     @. amp[begin+1:end-1] = 2*sqrt(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
#    #     @. phi[begin+1:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
#    #     amp[end] = abs(p_fft_real[end])
#    #     phi[end] = atan(zero(eltype(phi)), p_fft_real[end])
#    # else
#    #     # There are the same number of real and imaginary components (not
#    #     # counting the mean).
#    #     @. amp[begin+1:end] = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
#    #     @. phi[begin+1:end] = atan(p_fft_imag, p_fft_real)
#    # end

#    # # Find the sampling rate, which we can use later to find the frequency bins.
#    # fs = 1/timestep(ap)
#    # return PressureSpectrum(amp, phi, n, fs, starttime(ap))

#    rfft!(buf, p)
#    buf ./= n
#    return PressureSpectrum(buf, timestep(pth), starttime(pth))
#end

## function amplitude(ps::AbstractPressureSpectrum)
##     hc = halfcomplex(ps)
##     n = length(hc)

##     # Split the FFT output in half-complex format into mean, real, and imaginary
##     # components (returns views for the latter two).
##     p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

##     # output length.
##     n_real = length(p_fft_real)
##     m = 1 + n_real

##     # Now I want to get the amplitude and phase.
##     amp = similar(hc, m)
##     # Set the amplitude and phase for the mean component.
##     amp[begin] = abs(p_fft_mean)
##     if iseven(n)
##         # There is one more real component than imaginary component (not
##         # counting the mean).
##         @. amp[begin+1:end-1] = 2*sqrt(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
##         amp[end] = abs(p_fft_real[end])
##     else
##         # There are the same number of real and imaginary components (not
##         # counting the mean).
##         @. amp[begin+1:end] = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
##     end

##     return amp
## end

#@concrete struct PressureSpectrumAmplitude{T,IsEven,TData} <: AbstractVector{T} where {T,IsEven,TData<:AbstractVector{T}}
#    hc::TData
#    dt
#    t0
#end

#@inline function Base.size(psa::PressureSpectrumAmplitude)
#    return (size_amp_phase(psa),)
#end

#@inline function Base.getindex(psa::PressureSpectrumAmplitude, i::Int)
#    n = size_amp_phase(psa)
#    checkindex(Bool, 1:n, i) || throw(BoundsError(psa, i))

#    hc_mean, hc_real, hc_imag = split_hc_real_imag(halfcomplex(psa))
#    # n = length(psa.hc)
#    # Hmm... so I need to figure out which parts of hc I want.
#    if i == 1
#        amp = abs(hc_mean)
#    else
#        if iseven(n)
#            if i == n
#                # Nyquist frequency.
#                amp = abs(hc_real[i])
#            else
#                amp = 2*sqrt(hc_real[i]^2 + hc_imag[i]^2)
#            end
#        else
#            amp = 2*sqrt(hc_real[i]^2 + hc_imag[i]^2)
#        end
#    end

#    return amp
#end

#@concrete struct PressureSpectrumPhase{T, TData} <: AbstractVector{T} where {T,TData<:AbstractVector{T}}
#    hc::TData
#    fs
#    t0
#end

#@inline function Base.size(psa::PressureSpectrumAmplitude)
#    return (size_amp_phase(psa),)
#end

#@inline function Base.getindex(psa::PressureSpectrumAmplitude, i::Int)
#    n = size_amp_phase(psa)
#    checkindex(Bool, 1:n, i) || throw(BoundsError(psa, i))

#    hc_mean, hc_real, hc_imag = split_hc_real_imag(halfcomplex(psa))
#    # n = length(psa.hc)
#    # Hmm... so I need to figure out which parts of hc I want.
#    if i == 1
#        amp = abs(hc_mean)
#    else
#        if iseven(n)
#            if i == n
#                # Nyquist frequency.
#                amp = abs(hc_real[i])
#            else
#                amp = 2*sqrt(hc_real[i]^2 + hc_imag[i]^2)
#            end
#        else
#            amp = 2*sqrt(hc_real[i]^2 + hc_imag[i]^2)
#        end
#    end

#    return amp
#end

#function phase_t0(ps::AbstractPressureSpectrum)
#    hc = halfcomplex(ps)
#    n = length(hc)

#    # Split the FFT output in half-complex format into mean, real, and imaginary
#    # components (returns views for the latter two).
#    p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

#    # output length.
#    n_real = length(p_fft_real)
#    m = 1 + n_real

#    phi = similar(hc, m)
#    phi[begin] = atan(zero(eltype(phi)), p_fft_mean)
#    if iseven(n)
#        # There is one more real component than imaginary component (not
#        # counting the mean).
#        @. phi[begin+1:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
#        phi[end] = atan(zero(eltype(phi)), p_fft_real[end])
#    else
#        # There are the same number of real and imaginary components (not
#        # counting the mean).
#        @. phi[begin+1:end] = atan(p_fft_imag, p_fft_real)
#    end

#    return phi
#end

#function AcousticPressure(ps::AbstractPressureSpectrum)
#    amp = amplitude(ps)
#    phi = phase_t0(ps)
#    n = inputlength(ps)
#    T = promote_type(eltype(amp), eltype(phi))
#    p_fft = Vector{T}(undef, n)
#    _, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

#    n_real = length(p_fft_real)

#    p_fft[begin] = amp[begin]*cos(phi[begin])
#    # Both amp and phi always have the same length, and it's always one more
#    # than p_fft_real.
#    if mod(n, 2) == 0
#        # The length of p_fft_real is one greater than p_fft_imag.
#        # So the length of amp and phi are two greater than p_fft_imag.
#        @. p_fft_real[begin:end-1] = 0.5*amp[begin+1:end-1]*cos(phi[begin+1:end-1])
#        p_fft_real[end] = amp[end]*cos(phi[end])
#        @. p_fft_imag = 0.5*amp[begin+1:end-1]*sin(phi[begin+1:end-1])
#    else
#        # The length of p_fft_real is the same as p_fft_imag.
#        # So the length of amp and phi are one greater than p_fft_real and p_fft_imag.
#        @. p_fft_real = 0.5*amp[begin+1:end]*cos(phi[begin+1:end])
#        @. p_fft_imag = 0.5*amp[begin+1:end]*sin(phi[begin+1:end])
#    end

#    # So now let's do an inverse FFT.
#    p = copy(p_fft)
#    r2r!(p, HC2R)

#    # Now we just have to figure out dt and t0.
#    dt = 1/samplerate(ps)
#    t0 = starttime(ps)
#    return AcousticPressure(p, dt, t0)
#end

#abstract type AbstractNarrowbandSpectrum end

## @concrete struct NarrowbandSpectrum <: AbstractNarrowbandSpectrum
##     amp
##     ϕ
##     n
##     fs
##     t0
## end

#@concrete struct NarrowbandSpectrum <: AbstractNarrowbandSpectrum
#    hc
#    fs
#    t0
#end

#NarrowbandSpectrum(hc, fs) = NarrowbandSpectrum(hc, fs, zero(typeof(fs)))

## @inline inputlength(nbs::AbstractNarrowbandSpectrum) = nbs.n
## @inline samplerate(nbs::AbstractNarrowbandSpectrum) = nbs.fs
## @inline frequency(nbs::AbstractNarrowbandSpectrum) = rfftfreq(inputlength(nbs), samplerate(nbs))
## @inline amplitude(nbs::AbstractNarrowbandSpectrum) = nbs.amp
## @inline phase(nbs::AbstractNarrowbandSpectrum) = rem2pi.(nbs.ϕ .- 2 .* pi .* frequency(nbs) .* starttime(nbs), RoundNearest)
## @inline starttime(nbs::AbstractNarrowbandSpectrum) = nbs.t0
#@inline halfcomplex(nbs::AbstractNarrowbandSpectrum) = nbs.hc
#@inline inputlength(nbs::AbstractNarrowbandSpectrum) = length(halfcomplex(nbs))
#@inline samplerate(nbs::AbstractNarrowbandSpectrum) = nbs.fs
#@inline starttime(nbs::AbstractNarrowbandSpectrum) = nbs.t0
#@inline frequency(nbs::AbstractNarrowbandSpectrum) = rfftfreq(inputlength(nbs), samplerate(nbs))
## @inline amplitude(nbs::AbstractNarrowbandSpectrum) = nbs.amp
#@inline phase(nbs::AbstractNarrowbandSpectrum) = rem2pi.(phase_t0(nbs) .- 2 .* pi .* frequency(nbs) .* starttime(nbs), RoundNearest)

## function NarrowbandSpectrum(ap::AbstractAcousticPressure)
##     p = pressure(ap)
##     n = length(p)

##     # Get the FFT of the acoustic pressure. Divide by n since FFT computes an
##     # "unnormalized" FFT.
##     p_fft = rfft(p)./n

##     # Split the FFT output in half-complex format into mean, real, and imaginary
##     # components (returns views for the latter two).
##     p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

##     # output length.
##     n_real = length(p_fft_real)
##     m = 1 + n_real

##     # Now I want to get the amplitude and phase.
##     amp = similar(p, m)
##     ϕ = similar(p, m)

##     amp[begin] = p_fft_mean^2
##     ϕ[begin] = atan(zero(eltype(ϕ)), p_fft_mean)
##     if mod(n, 2) == 0
##         @. amp[begin+1:end-1] = 2*(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
##         @. ϕ[2:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
##         amp[end] = p_fft_real[end]^2
##         ϕ[end] = atan(zero(eltype(ϕ)), p_fft_real[end])
##     else
##         @. amp[begin+1:end] = 2*(p_fft_real^2 + p_fft_imag^2)
##         @. ϕ[begin+1:end] = atan(p_fft_imag, p_fft_real)
##     end

##     # Find the sampling rate, which we can use later to find the frequency bins.
##     fs = 1/timestep(ap)
##     return NarrowbandSpectrum(amp, ϕ, n, fs, starttime(ap))
## end

#function NarrowbandSpectrum(ap::AbstractAcousticPressure, buf=similar(pressure(ap)))
#    ps = PressureSpectrum(ap, buf)
#    return NarrowbandSpectrum(halfcomplex(ps), samplerate(ps), starttime(ps))
#end

#function amplitude(nbs::AbstractNarrowbandSpectrum)
#    hc = halfcomplex(nbs)
#    n = length(hc)

#    # Split the FFT output in half-complex format into mean, real, and imaginary
#    # components (returns views for the latter two).
#    p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

#    # output length.
#    n_real = length(p_fft_real)
#    m = 1 + n_real

#    # Now I want to get the amplitude and phase.
#    # amp = similar(p, m)
#    # ϕ = similar(p, m)
#    amp = similar(hc, m)

#    amp[begin] = p_fft_mean^2
#    # ϕ[begin] = atan(zero(eltype(ϕ)), p_fft_mean)
#    if mod(n, 2) == 0
#        @. amp[begin+1:end-1] = 2*(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
#        # @. ϕ[2:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
#        amp[end] = p_fft_real[end]^2
#        # ϕ[end] = atan(zero(eltype(ϕ)), p_fft_real[end])
#    else
#        @. amp[begin+1:end] = 2*(p_fft_real^2 + p_fft_imag^2)
#        # @. ϕ[begin+1:end] = atan(p_fft_imag, p_fft_real)
#    end

#    # # Find the sampling rate, which we can use later to find the frequency bins.
#    # fs = 1/timestep(ap)
#    # return NarrowbandSpectrum(amp, ϕ, n, fs, starttime(ap))
#    return amp
#end

#function phase_t0(nbs::AbstractNarrowbandSpectrum)
#    ps = PressureSpectrum(nbs)
#    return phase_t0(ps)
#end

#function PressureSpectrum(nbs::AbstractNarrowbandSpectrum)
#    ps = PressureSpectrum(halfcomplex(nbs), samplerate(nbs), starttime(nbs))
#    return ps
#end

##function PressureSpectrum(nbs::AbstractNarrowbandSpectrum)
##    # amplitude(nbs) = 2*(p_fft_real^2 + p_fft_imag^2)
##    # amplitude(ps) = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
##    # amp = 2*sqrt(amplitude(nbs)/2)
##    # amp = 2*sqrt(2)*sqrt(amplitude(nbs))/2
##    # amp = sqrt(2)*sqrt(amplitude(nbs))
##    #
##    n = inputlength(nbs)

##    # But for the mean component:
##    # amplitude(nbs)[1] = p_fft_mean^2
##    # amplitude(ps)[1] = p_fft_mean
##    nbs_amp = amplitude(nbs)
##    amp = similar(nbs_amp)

##    amp[begin] = sqrt(nbs_amp[begin])
##    if mod(n, 2) == 0
##        @. amp[begin+1:end-1] = sqrt(2*nbs_amp[begin+1:end-1])
##        amp[end] = sqrt(nbs_amp[end])
##    else
##        @. amp[begin+1:end] = sqrt(2*nbs_amp[begin+1:end])
##    end

##    phi = copy(nbs.ϕ)
##    fs = samplerate(nbs)
##    return PressureSpectrum(amp, phi, n, fs, starttime(nbs))
##end

#function AcousticPressure(nbs::AbstractNarrowbandSpectrum)
#    # amp = amplitude(nbs)
#    # ϕ = nbs.ϕ
#    # n = inputlength(nbs)
#    # T = promote_type(eltype(amp), eltype(ϕ))
#    # p_fft = Vector{T}(undef, n)
#    # _, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

#    # p_fft[begin] = sqrt(amp[begin])*cos(ϕ[begin])
#    # # Both amp and phi always have the same length, and it's always one more
#    # # than p_fft_real.
#    # if mod(n, 2) == 0
#    #     # The length of p_fft_real is one greater than p_fft_imag.
#    #     # So the length of amp and phi are two greater than p_fft_imag.
#    #     @. p_fft_real[begin:end-1] = 0.5*sqrt(2*amp[begin+1:end-1])*cos(ϕ[begin+1:end-1])
#    #     p_fft_real[end] = sqrt(amp[end])*cos(ϕ[end])
#    #     @. p_fft_imag = 0.5*sqrt(2*amp[begin+1:end-1])*sin(ϕ[begin+1:end-1])
#    # else
#    #     # The length of p_fft_real is the same as p_fft_imag.
#    #     # So the length of amp and phi are one greater than p_fft_real and p_fft_imag.
#    #     @. p_fft_real = 0.5*sqrt(2*amp[begin+1:end])*cos(ϕ[begin+1:end])
#    #     @. p_fft_imag = 0.5*sqrt(2*amp[begin+1:end])*sin(ϕ[begin+1:end])
#    # end

#    # # So now let's do an inverse FFT.
#    # p = copy(p_fft)
#    # r2r!(p, HC2R)

#    # # Now we just have to figure out dt and t0.
#    # dt = 1/samplerate(nbs)
#    # return AcousticPressure(p, dt, starttime(nbs))

#    ps = PressureSpectrum(nbs)
#    return AcousticPressure(ps)
#end

#function OASPL(ap::AbstractAcousticPressure)
#    p = pressure(ap)
#    n = length(p)
#    p_mean = sum(p)/n
#    msp = sum((p .- p_mean).^2)/n
#    return 10*log10(msp/p_ref^2)
#end

#function OASPL(nbs::AbstractNarrowbandSpectrum)
#    amp = amplitude(nbs)
#    msp = sum(amp[begin+1:end])
#    return 10*log10(msp/p_ref^2)
#end
