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

function OASPL(pbs::AbstractProportionalBandSpectrum)
    return 10*log10(sum(pbs)/p_ref^2)
end

function OASPL(pbs::LazyNBProportionalBandSpectrum)
    return OASPL(pbs.msp)
end

function OASPL(pbs::GenericLazyNBProportionalBandSpectrum)
    # GenericLazyNBProportionalBandSpectrum doesn't allow for a zero frequency, so no need to skip the first entry.
    msp = sum(pbs.msp_amp)
    return 10*log10(msp/p_ref^2)
end

function OASPL(pbs::LazyPBSProportionalBandSpectrum)
    return OASPL(pbs.pbs)
end
