abstract type AbstractAcousticPressure end

@concrete struct AcousticPressure <: AbstractAcousticPressure
    p
    dt
    t0
end

AcousticPressure(p, dt) = AcousticPressure(p, dt, zero(typeof(dt)))

@inline pressure(ap::AbstractAcousticPressure) = ap.p
@inline timestep(ap::AbstractAcousticPressure) = ap.dt
@inline starttime(ap::AbstractAcousticPressure) = ap.t0

@inline function time(ap::AbstractAcousticPressure)
    n = length(pressure(ap))
    return starttime(ap) .+ (0:n-1) .* timestep(ap)
end

"""
    split_hc_real_imag(hc::AbstractVector)

Split the output of a Fourier Transform in [halfcomplex
format](https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html) into
its mean, real, and imaginary parts.

Returns a length-three tuple with the mean (a scalar) first, then a `@view` of
the real components, and a `@view` of the imaginary components.
"""
function split_hc_real_imag(hc)
    n = length(hc)
    n_real = n >> 1 # number of real outputs, not counting the mean
    n_imag = ((n + 1) >> 1) - 1

    # The mean of the signal is in the first element of the output array.
    # Always real for a real-input FFT.
    hc_mean = hc[begin]
    # Then the real parts of the positive frequencies.
    hc_real = @view hc[begin+1:n_real+1]
    # Then the imaginary parts of the positive frequencies, which are "backwards."
    hc_imag = @view hc[end:-1:end-n_imag+1]

    return hc_mean, hc_real, hc_imag
end

abstract type AbstractPressureSpectrum end

# @concrete struct PressureSpectrum <: AbstractPressureSpectrum
#     amp
#     ϕ
#     n
#     fs
#     t0
# end

@concrete struct PressureSpectrum <: AbstractPressureSpectrum
    hc
    fs
    t0
end

PressureSpectrum(hc, fs) = PressureSpectrum(hc, fs, zero(typeof(fs)))

# @inline inputlength(ps::AbstractPressureSpectrum) = ps.n
# @inline samplerate(ps::AbstractPressureSpectrum) = ps.fs
# @inline starttime(ps::AbstractPressureSpectrum) = ps.t0
# @inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
# @inline amplitude(ps::AbstractPressureSpectrum) = ps.amp
# @inline phase(ps::AbstractPressureSpectrum) = rem2pi.(ps.ϕ .- 2 .* pi .* frequency(ps) .* starttime(ps), RoundNearest)
@inline halfcomplex(ps::AbstractPressureSpectrum) = ps.hc
@inline inputlength(ps::AbstractPressureSpectrum) = length(halfcomplex(ps))
@inline samplerate(ps::AbstractPressureSpectrum) = ps.fs
@inline starttime(ps::AbstractPressureSpectrum) = ps.t0
@inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
# @inline amplitude(ps::AbstractPressureSpectrum) = ps.amp
@inline phase(ps::AbstractPressureSpectrum) = rem2pi.(phase_t0(ps) .- 2 .* pi .* frequency(ps) .* starttime(ps), RoundNearest)

function PressureSpectrum(ap::AbstractAcousticPressure, buf=similar(pressure(ap)))
    p = pressure(ap)
    n = length(p)

    # # Get the FFT of the acoustic pressure. Divide by n since FFT computes an
    # # "unnormalized" FFT.
    # p_fft = rfft(p)./n

    # # Split the FFT output in half-complex format into mean, real, and imaginary
    # # components (returns views for the latter two).
    # p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

    # # output length.
    # n_real = length(p_fft_real)
    # m = 1 + n_real

    # # Now I want to get the amplitude and phase.
    # amp = similar(p, m)
    # phi = similar(p, m)
    # # Set the amplitude and phase for the mean component.
    # amp[begin] = abs(p_fft_mean)
    # phi[begin] = atan(zero(eltype(phi)), p_fft_mean)
    # if mod(n, 2) == 0
    #     # There is one more real component than imaginary component (not
    #     # counting the mean).
    #     @. amp[begin+1:end-1] = 2*sqrt(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
    #     @. phi[begin+1:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
    #     amp[end] = abs(p_fft_real[end])
    #     phi[end] = atan(zero(eltype(phi)), p_fft_real[end])
    # else
    #     # There are the same number of real and imaginary components (not
    #     # counting the mean).
    #     @. amp[begin+1:end] = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
    #     @. phi[begin+1:end] = atan(p_fft_imag, p_fft_real)
    # end

    # # Find the sampling rate, which we can use later to find the frequency bins.
    # fs = 1/timestep(ap)
    # return PressureSpectrum(amp, phi, n, fs, starttime(ap))

    rfft!(buf, p)
    buf ./= n
    return PressureSpectrum(buf, 1/timestep(ap), starttime(ap))
end

function amplitude(ps::AbstractPressureSpectrum)
    hc = halfcomplex(ps)
    n = length(hc)

    # Split the FFT output in half-complex format into mean, real, and imaginary
    # components (returns views for the latter two).
    p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

    # output length.
    n_real = length(p_fft_real)
    m = 1 + n_real

    # Now I want to get the amplitude and phase.
    amp = similar(hc, m)
    # Set the amplitude and phase for the mean component.
    amp[begin] = abs(p_fft_mean)
    if iseven(n)
        # There is one more real component than imaginary component (not
        # counting the mean).
        @. amp[begin+1:end-1] = 2*sqrt(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
        amp[end] = abs(p_fft_real[end])
    else
        # There are the same number of real and imaginary components (not
        # counting the mean).
        @. amp[begin+1:end] = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
    end

    return amp
end

function phase_t0(ps::AbstractPressureSpectrum)
    hc = halfcomplex(ps)
    n = length(hc)

    # Split the FFT output in half-complex format into mean, real, and imaginary
    # components (returns views for the latter two).
    p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

    # output length.
    n_real = length(p_fft_real)
    m = 1 + n_real

    phi = similar(hc, m)
    phi[begin] = atan(zero(eltype(phi)), p_fft_mean)
    if iseven(n)
        # There is one more real component than imaginary component (not
        # counting the mean).
        @. phi[begin+1:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
        phi[end] = atan(zero(eltype(phi)), p_fft_real[end])
    else
        # There are the same number of real and imaginary components (not
        # counting the mean).
        @. phi[begin+1:end] = atan(p_fft_imag, p_fft_real)
    end

    return phi
end

function AcousticPressure(ps::AbstractPressureSpectrum)
    amp = amplitude(ps)
    phi = phase_t0(ps)
    n = inputlength(ps)
    T = promote_type(eltype(amp), eltype(phi))
    p_fft = Vector{T}(undef, n)
    _, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

    n_real = length(p_fft_real)

    p_fft[begin] = amp[begin]*cos(phi[begin])
    # Both amp and phi always have the same length, and it's always one more
    # than p_fft_real.
    if mod(n, 2) == 0
        # The length of p_fft_real is one greater than p_fft_imag.
        # So the length of amp and phi are two greater than p_fft_imag.
        @. p_fft_real[begin:end-1] = 0.5*amp[begin+1:end-1]*cos(phi[begin+1:end-1])
        p_fft_real[end] = amp[end]*cos(phi[end])
        @. p_fft_imag = 0.5*amp[begin+1:end-1]*sin(phi[begin+1:end-1])
    else
        # The length of p_fft_real is the same as p_fft_imag.
        # So the length of amp and phi are one greater than p_fft_real and p_fft_imag.
        @. p_fft_real = 0.5*amp[begin+1:end]*cos(phi[begin+1:end])
        @. p_fft_imag = 0.5*amp[begin+1:end]*sin(phi[begin+1:end])
    end

    # So now let's do an inverse FFT.
    p = copy(p_fft)
    r2r!(p, HC2R)

    # Now we just have to figure out dt and t0.
    dt = 1/samplerate(ps)
    t0 = starttime(ps)
    return AcousticPressure(p, dt, t0)
end

abstract type AbstractNarrowbandSpectrum end

# @concrete struct NarrowbandSpectrum <: AbstractNarrowbandSpectrum
#     amp
#     ϕ
#     n
#     fs
#     t0
# end

@concrete struct NarrowbandSpectrum <: AbstractNarrowbandSpectrum
    hc
    fs
    t0
end

NarrowbandSpectrum(hc, fs) = NarrowbandSpectrum(hc, fs, zero(typeof(fs)))

# @inline inputlength(nbs::AbstractNarrowbandSpectrum) = nbs.n
# @inline samplerate(nbs::AbstractNarrowbandSpectrum) = nbs.fs
# @inline frequency(nbs::AbstractNarrowbandSpectrum) = rfftfreq(inputlength(nbs), samplerate(nbs))
# @inline amplitude(nbs::AbstractNarrowbandSpectrum) = nbs.amp
# @inline phase(nbs::AbstractNarrowbandSpectrum) = rem2pi.(nbs.ϕ .- 2 .* pi .* frequency(nbs) .* starttime(nbs), RoundNearest)
# @inline starttime(nbs::AbstractNarrowbandSpectrum) = nbs.t0
@inline halfcomplex(nbs::AbstractNarrowbandSpectrum) = nbs.hc
@inline inputlength(nbs::AbstractNarrowbandSpectrum) = length(halfcomplex(nbs))
@inline samplerate(nbs::AbstractNarrowbandSpectrum) = nbs.fs
@inline starttime(nbs::AbstractNarrowbandSpectrum) = nbs.t0
@inline frequency(nbs::AbstractNarrowbandSpectrum) = rfftfreq(inputlength(nbs), samplerate(nbs))
# @inline amplitude(nbs::AbstractNarrowbandSpectrum) = nbs.amp
@inline phase(nbs::AbstractNarrowbandSpectrum) = rem2pi.(phase_t0(nbs) .- 2 .* pi .* frequency(nbs) .* starttime(nbs), RoundNearest)

# function NarrowbandSpectrum(ap::AbstractAcousticPressure)
#     p = pressure(ap)
#     n = length(p)

#     # Get the FFT of the acoustic pressure. Divide by n since FFT computes an
#     # "unnormalized" FFT.
#     p_fft = rfft(p)./n

#     # Split the FFT output in half-complex format into mean, real, and imaginary
#     # components (returns views for the latter two).
#     p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

#     # output length.
#     n_real = length(p_fft_real)
#     m = 1 + n_real

#     # Now I want to get the amplitude and phase.
#     amp = similar(p, m)
#     ϕ = similar(p, m)

#     amp[begin] = p_fft_mean^2
#     ϕ[begin] = atan(zero(eltype(ϕ)), p_fft_mean)
#     if mod(n, 2) == 0
#         @. amp[begin+1:end-1] = 2*(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
#         @. ϕ[2:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
#         amp[end] = p_fft_real[end]^2
#         ϕ[end] = atan(zero(eltype(ϕ)), p_fft_real[end])
#     else
#         @. amp[begin+1:end] = 2*(p_fft_real^2 + p_fft_imag^2)
#         @. ϕ[begin+1:end] = atan(p_fft_imag, p_fft_real)
#     end

#     # Find the sampling rate, which we can use later to find the frequency bins.
#     fs = 1/timestep(ap)
#     return NarrowbandSpectrum(amp, ϕ, n, fs, starttime(ap))
# end

function NarrowbandSpectrum(ap::AbstractAcousticPressure, buf=similar(pressure(ap)))
    ps = PressureSpectrum(ap, buf)
    return NarrowbandSpectrum(halfcomplex(ps), samplerate(ps), starttime(ps))
end

function amplitude(nbs::AbstractNarrowbandSpectrum)
    hc = halfcomplex(nbs)
    n = length(hc)

    # Split the FFT output in half-complex format into mean, real, and imaginary
    # components (returns views for the latter two).
    p_fft_mean, p_fft_real, p_fft_imag = split_hc_real_imag(hc)

    # output length.
    n_real = length(p_fft_real)
    m = 1 + n_real

    # Now I want to get the amplitude and phase.
    # amp = similar(p, m)
    # ϕ = similar(p, m)
    amp = similar(hc, m)

    amp[begin] = p_fft_mean^2
    # ϕ[begin] = atan(zero(eltype(ϕ)), p_fft_mean)
    if mod(n, 2) == 0
        @. amp[begin+1:end-1] = 2*(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
        # @. ϕ[2:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
        amp[end] = p_fft_real[end]^2
        # ϕ[end] = atan(zero(eltype(ϕ)), p_fft_real[end])
    else
        @. amp[begin+1:end] = 2*(p_fft_real^2 + p_fft_imag^2)
        # @. ϕ[begin+1:end] = atan(p_fft_imag, p_fft_real)
    end

    # # Find the sampling rate, which we can use later to find the frequency bins.
    # fs = 1/timestep(ap)
    # return NarrowbandSpectrum(amp, ϕ, n, fs, starttime(ap))
    return amp
end

function phase_t0(nbs::AbstractNarrowbandSpectrum)
    ps = PressureSpectrum(nbs)
    return phase_t0(ps)
end

function PressureSpectrum(nbs::AbstractNarrowbandSpectrum)
    ps = PressureSpectrum(halfcomplex(nbs), samplerate(nbs), starttime(nbs))
    return ps
end

#function PressureSpectrum(nbs::AbstractNarrowbandSpectrum)
#    # amplitude(nbs) = 2*(p_fft_real^2 + p_fft_imag^2)
#    # amplitude(ps) = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
#    # amp = 2*sqrt(amplitude(nbs)/2)
#    # amp = 2*sqrt(2)*sqrt(amplitude(nbs))/2
#    # amp = sqrt(2)*sqrt(amplitude(nbs))
#    #
#    n = inputlength(nbs)

#    # But for the mean component:
#    # amplitude(nbs)[1] = p_fft_mean^2
#    # amplitude(ps)[1] = p_fft_mean
#    nbs_amp = amplitude(nbs)
#    amp = similar(nbs_amp)

#    amp[begin] = sqrt(nbs_amp[begin])
#    if mod(n, 2) == 0
#        @. amp[begin+1:end-1] = sqrt(2*nbs_amp[begin+1:end-1])
#        amp[end] = sqrt(nbs_amp[end])
#    else
#        @. amp[begin+1:end] = sqrt(2*nbs_amp[begin+1:end])
#    end

#    phi = copy(nbs.ϕ)
#    fs = samplerate(nbs)
#    return PressureSpectrum(amp, phi, n, fs, starttime(nbs))
#end

function AcousticPressure(nbs::AbstractNarrowbandSpectrum)
    # amp = amplitude(nbs)
    # ϕ = nbs.ϕ
    # n = inputlength(nbs)
    # T = promote_type(eltype(amp), eltype(ϕ))
    # p_fft = Vector{T}(undef, n)
    # _, p_fft_real, p_fft_imag = split_hc_real_imag(p_fft)

    # p_fft[begin] = sqrt(amp[begin])*cos(ϕ[begin])
    # # Both amp and phi always have the same length, and it's always one more
    # # than p_fft_real.
    # if mod(n, 2) == 0
    #     # The length of p_fft_real is one greater than p_fft_imag.
    #     # So the length of amp and phi are two greater than p_fft_imag.
    #     @. p_fft_real[begin:end-1] = 0.5*sqrt(2*amp[begin+1:end-1])*cos(ϕ[begin+1:end-1])
    #     p_fft_real[end] = sqrt(amp[end])*cos(ϕ[end])
    #     @. p_fft_imag = 0.5*sqrt(2*amp[begin+1:end-1])*sin(ϕ[begin+1:end-1])
    # else
    #     # The length of p_fft_real is the same as p_fft_imag.
    #     # So the length of amp and phi are one greater than p_fft_real and p_fft_imag.
    #     @. p_fft_real = 0.5*sqrt(2*amp[begin+1:end])*cos(ϕ[begin+1:end])
    #     @. p_fft_imag = 0.5*sqrt(2*amp[begin+1:end])*sin(ϕ[begin+1:end])
    # end

    # # So now let's do an inverse FFT.
    # p = copy(p_fft)
    # r2r!(p, HC2R)

    # # Now we just have to figure out dt and t0.
    # dt = 1/samplerate(nbs)
    # return AcousticPressure(p, dt, starttime(nbs))

    ps = PressureSpectrum(nbs)
    return AcousticPressure(ps)
end

function OASPL(ap::AbstractAcousticPressure)
    p = pressure(ap)
    n = length(p)
    p_mean = sum(p)/n
    msp = sum((p .- p_mean).^2)/n
    return 10*log10(msp/p_ref^2)
end

function OASPL(nbs::AbstractNarrowbandSpectrum)
    amp = amplitude(nbs)
    msp = sum(amp[begin+1:end])
    return 10*log10(msp/p_ref^2)
end
