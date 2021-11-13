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

"""
    W_A(nbs::AbstractNarrowbandSpectrum)

A-weight and return the amplitudes of `nbs`.
"""
function W_A(nbs::AbstractNarrowbandSpectrum)
    freq = frequency(nbs)
    amp = amplitude(nbs)
    amp .*= W_A.(freq)
    return amp
end
