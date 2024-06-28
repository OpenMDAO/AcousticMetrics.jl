"""
    dft_r2hc(x::AbstractVector)

Calculate the real-input discrete Fourier transform, returning the result in the "half-complex" format.

See
http://www.fftw.org/fftw3_doc/The-1d-Real_002ddata-DFT.html#The-1d-Real_002ddata-DFT
and http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html for
details.

Only use this for checking the derivatives of the FFT routines (should work fine, just slow).
"""
function dft_r2hc(x::AbstractVector)
    # http://www.fftw.org/fftw3_doc/The-1d-Real_002ddata-DFT.html#The-1d-Real_002ddata-DFT
    # http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html
    # So
    #
    #   * we don't need the imaginary part of y_0 (which will be the first element in y, say, i=1)
    #   * if n is even, we don't need the imaginary part of y_{n/2} (which would be i = n/2+1)
    #
    # Now, the order is supposed to be like this (r for real, i for imaginary):
    #   
    #   * r_0, r_1, r_2, r_{n/2}, i_{(n+1)/2-1}, ..., i_2, i_1
    #
    # But the docs say that they're still using the same old formula, which is:
    #
    #   Y_k = Σ_{j=0}^{n-1} X_j exp(-2*π*i*j*k/n)
    #
    # (where i is sqrt(-1)).
    n = length(x)
    xo = OffsetArray(x, 0:n-1)

    y = similar(x)
    yo = OffsetArray(y, 0:n-1)

    # Let's do k = 0 first.
    yo[0] = sum(xo)

    # Now go from k = 1 to n/2 for the real parts.
    T = eltype(x)
    for k in 1:n÷2
        yo[k] = zero(T)
        for j in 0:n-1
            # yo[k] += xo[j]*exp(-2*pi*sqrt(-1)*j*k/n)
            yo[k] += xo[j]*cos(-2*pi*j*k/n)
        end
    end

    # Now go from 1 to (n+1)/2-1 for the imaginary parts.
    for k in 1:(n+1)÷2-1
        yo[n-k] = zero(T)
        for j in 0:n-1
            yo[n-k] += xo[j]*sin(-2*pi*j*k/n)
        end
    end

    return y
end

"""
    dft_hc2r(x::AbstractVector)

Calculate the inverse discrete Fourier transform of a real-input DFT.

This is the inverse of `dft_r2hc`, except for a factor of `N`, where `N` is the length of the input (and output), since FFTW computes an "unnormalized" FFT.

See
http://www.fftw.org/fftw3_doc/The-1d-Real_002ddata-DFT.html#The-1d-Real_002ddata-DFT
and http://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html for
details.

Only use this for checking the derivatives of the FFT routines (should work fine, just slow).
"""
function dft_hc2r(x::AbstractVector)
    n = length(x)
    xo = OffsetArray(x, 0:n-1)

    y = zero(x)
    yo = OffsetArray(y, 0:n-1)

    j = 0
    for k in 0:n-1
        yo[k] += xo[j]
    end

    # So, I need this loop to get r_1 to r_{n÷2} and i_{(n+1)÷2-1} to i_1.
    # Let's say n is even.
    # Maybe 8.
    # So then n÷2 == 4 and (n+1)÷2-1 == 3.
    # So x0 looks like this:
    #
    #   r_0, r_1, r_2, r_3, r_4, i_3, i_2, i_1
    #
    # If n is odd, say, 9, then n÷2 == 4 and (n+1)÷2-1 == 4, and x0 looks like this:
    #
    #   r_0, r_1, r_2, r_3, r_4, i_4, i_3, i_2, i_1
    #
    for j in 1:(n-1)÷2
        rj = xo[j]
        ij = xo[n-j]
        for k in 0:n-1
            yo[k] += 2*rj*cos(2*pi*j*k/n) - 2*ij*sin(2*pi*j*k/n)
        end
    end

    if iseven(n)
        # Handle the Nyquist frequency.
        j = n÷2
        rj = xo[j]
        for k in 0:n-1
            yo[k] += rj*cos(2*pi*j*k/n)
        end
    end

    return y
end

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
