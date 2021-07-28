# `AcousticMetrics.jl`

## The Fourier Transform
`AcousticMetrics.jl` uses the FFTW library, a very popular
implementation of the fast Fourier transform (FFT) algorithm. The FFT is a
method of computing the discrete Fourier transform (DFT) that reduces the
computational complexity from ``n^2`` to ``n \log(n)``, where ``n`` is the length
of the input to the transform. FFTW's definition of the discrete Fourier
transform is
```math
  y_k = \sum_{j=0}^{n-1} x_j e^{-2 \pi \imath jk/n}
```
where ``\imath=\sqrt{-1}``.

The goal of a Fourier transform is to take a function (let's call it a "signal") and express it as a sum of
sinusoids. Let's imagine we have a very simple signal
```math
p(t) = A \sin(ωt+φ)
```
where we call ``A`` the amplitude, ``ω`` the frequency, and ``φ`` the phase. Say
we evaluate that function ``n`` times over a period ``T``, and assume that ``ω =
2πm/T``, i.e., that the period of our signal is some integer fraction of the
sampling period ``T``, since
```math
\frac{2 π}{ω} = \frac{2 π}{\frac{2πm}{T}} = \frac{T}{m}.
```
We'll refer to each of those samples as ``p_j = p(t_j)``, where ``t_j``
is the time at sample ``j`` and ``j`` is an index from ``0`` to ``n-1``. What is
the discrete Fourier transform of our signal ``p_j``? We should be able to
figure that out if we can express our signal ``p(t)`` as something that looks
like FFTW's definition of the DFT. How can we do that? Well, first we need to
remember that
```math
\sin(α+β) = \sin(α)\cos(β) + \cos(α)\sin(β)
```
which lets us rewrite our signal as
```math
p(t) = A \sin(ωt+φ) = A\left[ \sin(ωt)\cos(φ) + \cos(ωt)\sin(φ) \right] = A \cos(φ)\sin(ωt) + A\sin(φ)\cos(ωt).
```
Now, if we also remember that
```math
e^{ix} = \cos(x) + \imath \sin(x).
```
we can replace ``\sin(ωt)`` with
```math
\sin(ωt) = \frac{e^{\imath ωt} - e^{-\imath ωt}}{2\imath} = \frac{-\imath e^{\imath ωt} + \imath e^{-\imath ωt}}{2}
```
and ``\cos(ωt)`` with
```math
\cos(ωt) = \frac{e^{\imath ωt} + e^{-\imath ωt}}{2}.
```
Throw all that together and we get
```math
\begin{aligned}
p(t) &= A \sin(ωt+φ) = A \cos(φ)\left[ \frac{-\imath e^{\imath ωt} + \imath e^{-\imath ωt} }{2}\right] + A\sin(φ)\left[\frac{e^{\imath ωt} + e^{-\imath ωt}}{2}\right] \\
     &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{\imath ωt} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-\imath ωt}.
\end{aligned}
```
This is looking much closer to the definition of the DFT that we started with.
Next thing we need to do is realize that since we've sampled our signal ``p(t)``
at ``n`` different times, equally spaced, over a time period ``T``, then we
can replace ``t`` with
```math
t_j = j \frac{T}{n} = j Δt
```
where ``T/n`` is ``Δt``, the time step size. We've previously said that ``ω=\frac{2πm}{T}``, which implies that
```math
ω t_j = \left( \frac{2πm}{T} \right) \left(j \frac{T}{n} \right) = \frac{2πmj}{n}
```
So if we throw that in there, we find
```math
\begin{aligned}
p(t_j) = p_j &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{\imath ωt_j} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-\imath ωt_j} \\
             &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{\imath \frac{2πmj}{n}} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-\imath \frac{2πmj}{n}} \\
             &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath jm/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath jm/n}.
\end{aligned}
```

That's very close to the DFT definition. So what happens if we evaluate the DFT
of that last expression?
We'll call the discrete Fourier transform of the signal ``\hat{p}_k``. So...
```math
\begin{aligned}
  \hat{p}_k &= \sum_{j=0}^{n-1} p_j e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath jm/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath jm/n} \right) e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath j(m+k)/n} \right)
\end{aligned}
```
