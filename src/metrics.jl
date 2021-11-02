function spectrumfreq(n, dt)
    freq = 0:floor(Int, n/2)
    T = n*dt
    return freq./T
end

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

abstract type AbstractPressureSpectrum end

@concrete struct PressureSpectrum <: AbstractPressureSpectrum
    n
    fs
    amp
    ϕ
end

@inline inputlength(ps::AbstractPressureSpectrum) = ps.n
@inline samplerate(ps::AbstractPressureSpectrum) = ps.fs
@inline frequency(ps::AbstractPressureSpectrum) = rfftfreq(inputlength(ps), samplerate(ps))
@inline amplitude(ps::AbstractPressureSpectrum) = ps.amp
@inline phase(ps::AbstractPressureSpectrum) = ps.ϕ

function PressureSpectrum(ap::AbstractAcousticPressure)
    p = pressure(ap)
    n = length(p)

    # output length.
    m = 1 + floor(Int, n/2)

    # Get the FFT of the acoustic pressure. FFTW computes "unnormalized" FFTs,
    # so we need to divide by the length of the input array (the number of
    # observer times).
    p_fft = rfft(p)./n

    # The mean of the signal is in the first element of the output array.
    # Always real for a real-input FFT.
    p_fft_mean = p_fft[begin]
    # Then the real parts of the positive frequencies.
    p_fft_real = @view p_fft[begin+1:m]
    # Then the imaginary parts of the negative frequencies, which are
    # "backwards" and have the opposite sign of the coresponding positive
    # frequency. But that's not working, sad. Seems like the negative is wrong.
    # Hmm... Maybe the second half of the rfft output are actually the positive
    # frequencies? Oh, shoot, that's right. Yay!
    p_fft_imag = @view p_fft[end:-1:m+1]
    # p_fft_imag .*= -1

    # Now I want to get the amplitude and phase.
    amp = similar(p, m)
    ϕ = similar(p, m)
    amp[begin] = p_fft_mean
    ϕ[begin] = zero(eltype(ϕ))  # imaginary component is zero, so atan(0) == 0.
    if mod(n, 2) == 0
        @. amp[begin+1:end-1] = 2*sqrt(p_fft_real[begin:end-1]^2 + p_fft_imag^2)
        amp[end] = p_fft_real[end]  # Is this really right? Apparently it is, but I don't know why. I should look at that.
        @. ϕ[begin+1:end-1] = atan(p_fft_imag, p_fft_real[begin:end-1])
        ϕ[end] = zero(eltype(ϕ))  # imaginary component is zero, so atan(0) == 0.
    else
        @. amp[begin+1:end] = 2*sqrt(p_fft_real^2 + p_fft_imag^2)
        @. ϕ[begin+1:end] = atan(p_fft_imag, p_fft_real)
    end

    # Find the sampling rate, which we can use later to find the frequency bins.
    fs = 1/timestep(ap)
    return PressureSpectrum(n, fs, amp, ϕ)
end

function AcousticPressure(ps::AbstractPressureSpectrum)
    amp = amplitude(ps)
    ϕ = phase(ps)
    n = inputlength(ps)
    m = length(amp)  # this is the number of FFT amplitude/phase outputs...
    p_fft = zeros(eltype(amp), n)
    p_fft[begin] = amp[begin]
    if mod(n, 2) == 0
        @. p_fft[begin+1:m] = 0.5*amp[begin+1:end-1]*cos(ϕ[begin+1:end-1])
        # @. p_fft[begin+1:end-1] = 0.5*amp[begin+1:end-1]*cos(ϕ[begin+1:end-1])
    else
        @. p_fft[begin+1:m] = 0.5*amp[begin+1:end]*cos(ϕ[begin+1:end])
        @. p_fft[end:-1:m+1] = 0.5*amp[begin+1:end]*sin(ϕ[begin+1:end])
    end

end

abstract type AbstractNarrowbandSpectrum end

@concrete struct NarrowbandSpectrum <: AbstractNarrowbandSpectrum
    freq
    amp
    ϕ
end

@inline frequency(nbs::AbstractNarrowbandSpectrum) = nbs.freq
@inline amplitude(nbs::AbstractNarrowbandSpectrum) = nbs.amp
@inline phase(nbs::AbstractNarrowbandSpectrum) = nbs.ϕ

function NarrowbandSpectrum(ap::AbstractAcousticPressure)
    p = pressure(ap)
    n = length(p)

    # Get the FFT of the acoustic pressure. FFTW computes "unnormalized" FFTs,
    # so we need to divide by the length of the input array (the number of
    # observer times).
    p_fft = rfft(p)./n

    # The mean of the signal is in the first element of the output array.
    p_fft_mean = p_fft[1]
    # Then the real parts.
    p_fft_real = @view p_fft[2:floor(Int, n/2)+1]
    # Then the imaginary parts, which are "backwards."
    p_fft_imag = @view p_fft[end:-1:floor(Int, n/2)+2]

    # What's the size of the output nbs going to be? It will be the mean
    # component (1) + floor(n/2)
    nbs_length = 1 + floor(Int, n/2)
    amp = similar(p, nbs_length)
    phase = similar(p, nbs_length)

    amp[1] = p_fft_mean^2
    # If the input length is even, there will be one more real component than
    # the imaginary. The missing imaginary part for the even-length case is
    # always zero. If the input length is odd, then the real and imaginary
    # components will be the same length.
    if mod(n, 2) == 0
        amp[2:end-1] .= p_fft_real[begin:end-1].^2 .+ p_fft_imag.^2
        amp[end] = p_fft_real[end]^2
        phase[2:end-1] .= atan.(p_fft_imag, p_fft_real[begin:end-1])
        phase[end] = zero(eltype(phase))  # imaginary component is zero, so atan(0) == 0.
    else
        amp[2:end] .= p_fft_real.^2 .+ p_fft_imag.^2
        phase[2:end] .= atan.(p_fft_imag, p_fft_real)
    end

    # Now we need to multipy the amplitude (except the mean part) by two to get
    # the energy associated with the other half of the spectrum.
    amp[2:end] .*= 2.0

    # Also need the frequency.
    dt = timestep(ap)
    # freq = spectrumfreq(n, dt)
    freq = rfftfreq(n, dt)

    return NarrowbandSpectrum(freq, amp, phase)
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
    msp = sum(amp[2:end])
    return 10*log10(msp/p_ref^2)
end
