struct RFFTCache{TVal,TJac}
    val::TVal
    jac::TJac
end

function RFFTCache(::Type{V}, M, N) where {V}
    val = Vector{V}(undef, M)
    jac = Matrix{V}(undef, M, N)
    return RFFTCache(val, jac)
end

function RFFTCache(d)
    M = length(d)
    T = eltype(d)
    N = ForwardDiff.npartials(T)
    V = ForwardDiff.valtype(T)
    return RFFTCache(V, M, N)
end

"""
    rfft!(y, x, cache=nothing)

    Calculate the real-input FFT of `x` and store the result in half-complex format in `y`.

    Just a wrapper of `FFTW.r2r!(y, FFTW.R2HC)`. The `cache` argument is
    optional and not used, and is included to keep the function signiture the
    same as the method that takes `Vector`s of `Dual`s.
"""
function rfft!(y, x, cache=nothing)
    y .= x
    r2r!(y, R2HC)
    return nothing
end

function rfft!(dout::AbstractVector{ForwardDiff.Dual{T,V,N}}, d::AbstractVector{ForwardDiff.Dual{T,V,N}}, cache=RFFTCache(V,length(d),N)) where {T,V,N}
    # N is the number of parameters we're taking the derivative wrt.
    # M is the number of inputs to (and outputs of) the FFT.
    M = length(d)
    # I should check that dout and d are the same size.
    ldout = length(dout)
    M == ldout || throw(DimensionMismatch("dout and d should have the same length, but have $ldout and $M, resp."))

    # But now I'd need to be able to pass that to the FFTW library. Will that
    # work? No, because it's a dual number. Bummer. So I'll just have to have a
    # working array, I guess.
    cache.val .= ForwardDiff.value.(d)
    r2r!(cache.val, R2HC)

    # Now I want to do the Jacobian.
    for i in 1:N
        for j in 1:M
            cache.jac[j, i] = ForwardDiff.partials(d[j], i)
        end
    end
    r2r!(cache.jac, R2HC, 1)

    # Now I should be able to set dout.
    for j in 1:M
        dout[j] = ForwardDiff.Dual{T}(cache.val[j], ForwardDiff.Partials(NTuple{N,V}(cache.jac[j,:])))
    end

    return nothing
end

function rfft(x)
    y = similar(x)
    rfft!(y, x)
    return y
end

"""
    irfft!(y, x, cache=nothing)

    Calculate the inverse FFT of `x` and store the result in in `y`, where `x` is in the half-complex format.

    Just a wrapper of `FFTW.r2r!(y, FFTW.HC2R)`. The `cache` argument is
    optional and not used, and is included to keep the function signiture the
    same as the method that takes `Vector`s of `Dual`s.
"""
function irfft!(y, x, cache=nothing)
    y .= x
    r2r!(y, HC2R)
    return nothing
end

function irfft!(dout::AbstractVector{ForwardDiff.Dual{T,V,N}}, d::AbstractVector{ForwardDiff.Dual{T,V,N}}, cache=RFFTCache(V,length(d),N)) where {T,V,N}
    # N is the number of parameters we're taking the derivative wrt.
    # M is the number of inputs to (and outputs of) the FFT.
    M = length(d)
    # I should check that dout and d are the same size.
    ldout = length(dout)
    M == ldout || throw(DimensionMismatch("dout and d should have the same length, but have $ldout and $M, resp."))

    # But now I'd need to be able to pass that to the FFTW library. Will that
    # work? No, because it's a dual number. Bummer. So I'll just have to have a
    # working array, I guess.
    cache.val .= ForwardDiff.value.(d)
    r2r!(cache.val, HC2R)

    # Now I want to do the Jacobian.
    for i in 1:N
        for j in 1:M
            cache.jac[j, i] = ForwardDiff.partials(d[j], i)
        end
    end
    r2r!(cache.jac, HC2R, 1)

    # Now I should be able to set dout.
    for j in 1:M
        dout[j] = ForwardDiff.Dual{T}(cache.val[j], ForwardDiff.Partials(NTuple{N,V}(cache.jac[j,:])))
    end

    return nothing
end

function irfft(x)
    y = similar(x)
    irfft!(y, x)
    return y
end

# Wish this was implemented in FFTW.jl. Actually, I think it may be. Ah, but
# only for the real-input FFT that returns a complex array, which I don't want.
# Sad. Oh, wait: but are the frequencies at fftfreq? I'll need to look at that
# again. Need to write that up once I figure it out.
function r2rfftfreq(n, d=1.0)
    # http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
    freq = vcat(0:floor(Int, n/2), floor(Int, (n+1)/2)-1:-1:1)
    # Get the period.
    T = n*d
    return freq./T
end
