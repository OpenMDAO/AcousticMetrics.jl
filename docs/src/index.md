# `AcousticMetrics.jl`

## The Fourier Transform
`AcousticMetrics.jl` uses the FFTW library, a very popular
implementation of the fast Fourier transform (FFT) algorithm. The FFT is a
method of computing the discrete Fourier transform (DFT) that reduces the
computational complexity from ``n^2`` to ``n \log(n)``, where ``n`` is the length
of the input to the transform. [FFTW's definition of the discrete Fourier
transform](http://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html#The-1d-Discrete-Fourier-Transform-_0028DFT_0029) is
```math
  y_k = \sum_{j=0}^{n-1} x_j e^{-2 \pi \imath jk/n}
```
where ``\imath=\sqrt{-1}``.

The goal of a Fourier transform is to take a function (let's call it a "signal") and express it as a sum of
sinusoids. Let's imagine we have a very simple signal
```math
p(t) = A \sin(ωt+φ)
```
where we call ``A`` the amplitude, ``ω`` the frequency, and ``φ`` the phase.
`AcousticAnalogies.jl` is interested in (suprise!) acoustics, which are real
numbers, so we'll assume that ``A``, ``ω``, and ``φ`` are all real. Say we
evaluate that function ``n`` times over a period ``T``, and assume that ``ω =
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
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] e^{-2π\imath j(m+k)/n} \right) \\
            &=\frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] \sum_{j=0}^{n-1} e^{-2π\imath j(m+k)/n} 
\end{aligned}
```
Pretty close now. Let's think about those two summations in the last
expression. First assume that ``m - k = q \ne 0``, where ``m`` and ``k`` are
both integers. Then the first sum would be
```math
  \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} = \sum_{j=0}^{n-1} e^{2π\imath jq/n}.
```
That's a signal that has period
```math
\frac{2π}{2πq/n} = n/q
```
that we're sampling ``n`` times. So we're sampling a sinusoid an integer number
of times over its period, and summing it up. That will give us... zero. Same thing will
happen to the second sum if ``m+k=r \ne 0``: we'll also get zero. So now we just
have to figure out what happens when ``m - k = 0`` and ``m + k = 0``, i.e., when
``k ± m``. Let's try ``k = m`` first. The first sum will be
of times, 
```math
  \sum_{j=0}^{n-1} e^{2π\imath j(m-m)/n} = \sum_{j=0}^{n-1} e^{0} = \sum_{j=0}^{n-1} 1 = n
```
and, from the previous argument, we know the second sum will be 0, since
``m+k=2m \ne 0``. For ``k = -m``, the first sum will be zero, since ``m - -m =
2m \ne 0``, and the second sum will be
```math
  \sum_{j=0}^{n-1} e^{2π\imath j(m-m))/n} = n
```
again.

Great! So now we finally can write down the DFT of our example signal
```math
p(t) = A \sin(ωt+φ) = A \sin\left(\left[\frac{2πm}{T}\right]t+φ\right),
```
which is (wish I could figure out how to do the `cases` LaTeX environment)...
```math
\begin{aligned}
  \hat{p}_k &= \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] \sum_{j=0}^{n-1} e^{-2π\imath j(m+k)/n} \\
  \hat{p}_m & = \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```

What if we would have used `\cos` instead of `\sin` in our example signal? Well,
we have the identity
```math
\cos(ωt + θ) = \sin(ωt + θ + π/2).
```
So to find the DFT of the signal
```math
p(t) = A \cos(ωt+θ)
```
we just need to make the substitution $φ = θ + π/2$ in our original result,
which would give
```math
\begin{aligned}
  \hat{p}_m & = \frac{A}{2}\left[\sin(θ+π/2) - \imath \cos(θ+π/2) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\sin(θ+π/2) + \imath \cos(θ+π/2) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise},
\end{aligned}
```
or
```math
\begin{aligned}
  \hat{p}_m & = \frac{A}{2}\left[\cos(θ) + \imath \sin(θ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(θ) - \imath \sin(θ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```

We're almost ready to compare our example signal to the output of the FFTW
library. The last thing we need to think about is how FFTW's output is ordered.
FFT libraries have different conventions, but [here is what FFTW does](http://www.fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html#The-1d-Discrete-Fourier-Transform-_0028DFT_0029):
> Note also that we use the standard “in-order” output ordering—the k-th output
> corresponds to the frequency k/n (or k/T, where T is your total sampling
> period). For those who like to think in terms of positive and negative
> frequencies, this means that the positive frequencies are stored in the first
> half of the output and the negative frequencies are stored in backwards order
> in the second half of the output. (The frequency -k/n is the same as the
> frequency (n-k)/n.)
So for our original example signal
```math
\begin{aligned}
  p(t) &= A \sin(ωt+φ) \\
  \hat{p}_m & = \frac{A}{2}\left[\sin(φ) - \imath \cos(φ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\sin(φ) + \imath \cos(φ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```
we would expect `\hat{p}_m` to appear in the `m+1`th position (since we're counting from zero), and `\hat{p}_{-m}` to appear in the
