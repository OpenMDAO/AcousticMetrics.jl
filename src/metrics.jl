abstract type AbstractAcousticPressure end

@concrete struct AcousticPressure <: AbstractAcousticPressure
    p
    dt
end

@inline pressure(ap::AbstractAcousticPressure) = ap.p
@inline timestep(ap::AbstractAcousticPressure) = ap.dt

@inline function time(ap::AbstractAcousticPressure)
    n = length(pressure(ap))
    return (0:n-1) .* timestep(ap)
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
    freq = rfftfreq(n, dt)[1:nbs_length]

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
