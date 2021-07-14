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
we evaluate that function ``n`` times over a period ``2π/ω`` and refer to each
of those samples as ``p_j = p(t_j)``, where ``t_j`` is the time at sample ``j``
and ``j`` is an index from ``0`` to ``n-1``. What is the discrete Fourier
transform of our signal ``p_j``? We should be able to figure that out if we can
express our signal ``p(t)`` as something that looks like FFTW's definition of
the DFT. How can we do that? Well, first we need to remember that
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
at ``n`` different times, equally spaced, over a time period ``2π/ω``, then we
can replace ``t`` with
```math
t_j = j \frac{\frac{2π}{ω}}{n} = j \frac{T}{n} = j Δt.
```
The quantity ``T = \frac{2π}{ω}`` is the period, i.e. the length of time over
which we've decided to sample our signal, and so ``T/n`` is ``Δt``, the
time step size. And of course then ``ω t_j = j \frac{2π}{n}``, so let's throw
that in there:
```math
\begin{aligned}
p(t_j) = p_j &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{\imath ωt_j} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-\imath ωt_j} \\
             &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{\imath j \frac{2π}{n}} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-\imath j \frac{2π}{n}} \\
             &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath j/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath j/n}.
\end{aligned}
```

That's very close to the DFT definition. So what happens if we evaluate the DFT
of that last expression?
We'll call the discrete Fourier transform of the signal ``\hat{p}_k``. So...
```math
\begin{aligned}
  \hat{p}_k &= \sum_{j=0}^{n-1} p_j e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath j/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath j/n} \right) e^{-2 \pi \imath jk/n}.
\end{aligned}
```
(Now I've just realized that I want to change the definition of ``ω`` slightly.
I want it to be some **multiple** of ``2π/T``, not just ``2π/T``. Hopefully
that's not too hard.)
