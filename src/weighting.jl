"""
    W_A(f::AbstractFloat)

Calculate the A-weighting factor for a frequency `f` in Hertz.

Taken from the ANOPP2 Acoustics Analysis API Reference Manual.
"""
function W_A(f)
    f_1 = 20.598997
    f_2 = 107.65265
    f_3 = 737.86233
    f_4 = 12194.217
    f_5 = 158.48932
    f_6 = 79919.29
    f_7 = 1345600.0
    f_8 = 1037918.48
    f_9 = 9837328.0
    K_1 = 2.242881e16
    K_2 = 1.025119
    K_3 = 1.562339
    K_4 = 14500.0
    K_5 = 1080768.18
    K_6 = 11723776.0

    W_C = (K_1*f^4) / ((f^2 + f_1^2)^2*(f^2 + f_4^2)^2)
    w_a = (K_3*f^4*W_C) / ((f^2 + f_2^2)*(f^2 + f_3^2))

    return w_a
end

function W_A2(f)
    R_A = (12194^2 * f^4) / ( (f^2 + 20.6^2) * sqrt( (f^2 + 107.7^2) * (f^2 + 737.9^2) ) * (f^2 + 12194^2))
    R_A = (12194^2 * f^4) / ( (f^2 + 20.6^2) * sqrt( (f^2 + 107.7^2) * (f^2 + 737.9^2) ) * (f^2 + 12194^2))

    # A(f) = 20*log10(R_A(f)) - 20*log10(R_A(1000))
    #      = 10*log10(R_A(f)^2) - 10*log10(R_A(1000)^2)
    # msp_A(f) = A*(f)*msp(f)
    # spl = 10*log10(msp/pref^2)
    # spl_A = 10*log10(A*msp/pref^2) = 10*log10(msp/pref^2) + 10*log10(A)
end

"""
    a_weight!(sm::AbstractNarrowbandSpectrum)

A-weight a narrowband spectrum in place, returning the weighted spectrum.
"""
function a_weight! end

function a_weight!(sm::AbstractNarrowbandSpectrum{false})
    m = inputlength(sm)
    hc = halfcomplex(sm)
    freqs = frequency(sm)
    n = length(sm)
    @inbounds begin
        hc[1] *= sqrt(W_A(freqs[1]))
        for i in 2:n
            w = sqrt(W_A(freqs[i]))
            hc[i] *= w
            hc[m-i+2] *= w
        end
    end
    return sm
end

function a_weight!(sm::AbstractNarrowbandSpectrum{true})
    m = inputlength(sm)
    hc = halfcomplex(sm)
    freqs = frequency(sm)
    n = length(sm)
    @inbounds begin
        hc[1] *= sqrt(W_A(freqs[1]))
        for i in 2:n-1
            w = sqrt(W_A(freqs[i]))
            hc[i] *= w
            hc[m-i+2] *= w
        end
        hc[n] *= sqrt(W_A(freqs[n]))
    end

    return sm
end

"""
    a_weight(sm::AbstractNarrowbandSpectrum)

A-weight and return a narrowband spectrum without modifying the original input spectrum `sm`.
"""
function a_weight(sm::AbstractNarrowbandSpectrum) end

function a_weight(sm::AbstractNarrowbandSpectrum{false})
    m = inputlength(sm)
    hc = halfcomplex(sm)
    freqs = frequency(sm)
    n = length(sm)
    # Create an output sm.
    # Will this work?
    smout = similar(sm)
    hcout = halfcomplex(smout)
    @inbounds begin
        hcout[1] = hc[1]*sqrt(W_A(freqs[1]))
        for i in 2:n
            w = sqrt(W_A(freqs[i]))
            hcout[i] = hc[i]*w
            hcout[m-i+2] = hc[m-i+2]*w
        end
    end

    return smout
end

function a_weight(sm::AbstractNarrowbandSpectrum{true})
    m = inputlength(sm)
    hc = halfcomplex(sm)
    freqs = frequency(sm)
    n = length(sm)
    # Create an output sm.
    # Will this work?
    smout = similar(sm)
    hcout = halfcomplex(smout)
    @inbounds begin
        hcout[1] = hc[1]*sqrt(W_A(freqs[1]))
        for i in 2:n-1
            w = sqrt(W_A(freqs[i]))
            hcout[i] = hc[i]*w
            hcout[m-i+2] = hc[m-i+2]*w
        end
        hcout[n] = hc[n]*sqrt(W_A(freqs[n]))
    end

    return smout
end

function a_weight!(pbs::Union{ProportionalBandSpectrum,ProportionalBandSpectrumWithTime})
    cbands = center_bands(pbs)
    scaler = freq_scaler(pbs)
    msp = pbs.pbs
    @inbounds begin
        for i in eachindex(msp)
            freq = cbands[i]*scaler
            msp[i] *= W_A(freq)
        end
    end
    return pbs
end

function a_weight(pbs::Union{ProportionalBandSpectrum,ProportionalBandSpectrumWithTime})
    pbs_out = similar(pbs)
    cbands = center_bands(pbs_out)
    scaler = freq_scaler(pbs_out)
    msp = pbs.pbs
    msp_out = pbs_out.pbs
    @inbounds begin
        for i in eachindex(msp)
            freq = cbands[i]*scaler
            msp_out[i] = W_A(freq)*msp[i]
        end
    end
    return pbs_out
end

function a_weight!(pbs::LazyPBSProportionalBandSpectrum)
    a_weight!(pbs.pbs)
    return pbs
end

function a_weight(pbs::LazyPBSProportionalBandSpectrum)
    return lazy_pbs(a_weight(pbs.pbs), pbs.cbands)
end

function a_weight!(pbs::GenericLazyNBProportionalBandSpectrum)
    freqs = frequency_nb(pbs)
    msp = msp_amplitude(pbs)
    @inbounds begin
        for i in eachindex(msp)
            msp[i] *= W_A(freqs[i])
        end
    end
    return pbs
end

function a_weight(pbs::GenericLazyNBProportionalBandSpectrum)
    freqs = frequency_nb(pbs)
    msp = msp_amplitude(pbs)
    msp_out = similar(msp)
    @inbounds begin
        for i in eachindex(msp)
            msp_out[i] = W_A(freqs[i])*msp[i]
        end
    end
    NO = octave_fraction(pbs)
    IsTonal = istonal(pbs)
    f1_nb = startfrequency(pbs)
    df_nb = frequencystep(pbs)
    cbands = center_bands(pbs)
    return GenericLazyNBProportionalBandSpectrum{NO,IsTonal}(f1_nb, df_nb, msp_out, cbands)
end

function a_weight!(pbs::LazyNBProportionalBandSpectrum)
    a_weight!(pbs.msp)
    return pbs
end

function a_weight(pbs::LazyNBProportionalBandSpectrum)
    return lazy_pbs(a_weight(pbs.msp), center_bands(pbs))
end

# """
#     W_A(nbs::AbstractNarrowbandSpectrum)

# A-weight and return the amplitudes of `nbs`.
# """
# function W_A(nbs::AbstractNarrowbandSpectrum)
#     freq = frequency(nbs)
#     amp = amplitude(nbs)
#     amp .*= W_A.(freq)
#     return amp
# end
