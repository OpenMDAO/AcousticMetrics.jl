### What if the signal frequency isn't a multiple of the sampling frequency?
```math
p(t) = A \cos(ωt+φ)
```
Say we evaluate that function ``n`` times over a period ``T``, just like before.
But this time we will assume that ``ω = 2π(m+a)/T`` where ``0 \lt a \lt 1``, i.e., that the period of our signal is *not* some integer fraction of the sampling period ``T``.
What will the Fourier transform of that be?

We can reuse a bunch of our previous work.
This expression for the signal ``p(t)`` still applies:
```math
p(t) = \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{\imath ωt} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-\imath ωt}
```
But now we just need to substitute our new expression for ``ω t_j``,
```math
ω t_j = \left( \frac{2π(m+a)}{T} \right) \left(j \frac{T}{n} \right) = \frac{2π(m+a)j}{n},
```
which will give us
```math
p(t_j) = p_j = \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath j(m+a)/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath j(m+a)/n}
```
Now, if we do the FFT:
```math
\begin{aligned}
  \hat{p}_k &= \sum_{j=0}^{n-1} p_j e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(m+a)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(m+a)/n} \right) e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(m+a-k)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(m+a+k)/n} \right) \\
            &=\frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{2π\imath j(m+a-k)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{-2π\imath j(m+a+k)/n} \\
            &=\frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} e^{2π\imath aj/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{-2π\imath j(m+k)/n}  e^{2π\imath aj/n}
\end{aligned}
```

