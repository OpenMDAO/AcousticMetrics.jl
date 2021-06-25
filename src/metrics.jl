function nbs_from_apth(p, dt=one(eltype(p)))
    n = length(p)
    # @assert length(t) == num_obs_times

    # What's the size of the output nbs going to be? It will be the mean
    # component (1) + floor(n/2)
    nbs_length = 1 + floor(Int, n/2)
    nbs = similar(p, nbs_length)

    # Get the FFT of the acoustic pressure. FFTW computes "unnormalized" FFTs,
    # so we need to divide by the length of the input array (the number of
    # observer times).
    p_fft = rfft(p)./n

    # Get the magnitude of the discrete Fourier transform of the acoustic
    # pressure. The mean of the pressure signal is in the first element of the
    # output array.
    p_fft_mean = p_fft[1]
    # Then the real parts.
    p_fft_real = @view p_fft[2:floor(Int, n/2)+1]
    # Then the imaginary parts, which are "backwards."
    p_fft_imag = @view p_fft[end:-1:floor(Int, n/2)+2]

    nbs[1] = p_fft_mean^2
    # If the input length is even, there will be one more real component than
    # the imaginary. The missing imaginary part for the even-length case is
    # always zero. If the input length is odd, then the real and imaginary
    # components will be the same length.
    if mod(n, 2) == 0
        nbs[2:end-1] .= p_fft_real[begin:end-1].^2 .+ p_fft_imag.^2
        nbs[end] = p_fft_real[end]^2
    else
        nbs[2:end] .= p_fft_real.^2 .+ p_fft_imag.^2
    end

    # Now we need to multipy everything except the mean part by two to get the
    # energy associated with the other half of the spectrum.
    nbs[2:end] .*= 2.0

    # # First, get the "FFT frequency," which is the number of cycles per inputs.
    # # So, a frequency of 3 means 3 cycles over n samples. But I
    # # don't really want to divide by n, so I wont.
    # freq = 0:floor(Int, n/2)
    # # Now, to get the "real" frequency in cycles per s, I'd multiply by
    # # (n/<time length of the signal>. But I didn't divide by
    # # n, so I just need to divide by the time length of the signal.
    # T = n*dt
    # freq = freq ./ T
    freq = rfftfreq(n, dt)[1:floor(Int, n/2)+1]

    return freq, nbs
end

function oaspl_from_apth(p)
    n = length(p)
    p_mean = sum(p)/n
    msp = sum((p .- p_mean).^2)/n
    return 10*log10(msp/p_ref^2)
end

function oaspl_from_nbs(nbs)
    # n = length(nbs)
    # return sum(nbs[2:end])/n
    msp = sum(nbs[2:end])
    return 10*log10(msp/p_ref^2)
end
