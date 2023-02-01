```@meta
CurrentModule = AMDocs
```
# `AcousticMetrics.jl`

## The Fourier Transform
### The Basics
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
p(t) = A \cos(ωt+φ)
```
where we call ``A`` the amplitude, ``ω`` the frequency, and ``φ`` the phase.
`AcousticAnalogies.jl` is interested in (surprise!) acoustics, which are real
numbers, so we'll assume that ``A``, ``ω``, and ``φ`` are all real. Say we
evaluate that function ``n`` times over a period ``T``, and assume that ``ω =
2πm/T``, i.e., that the period of our signal is some integer fraction of the
sampling period ``T``, since
```math
\frac{2 π}{ω} = \frac{2 π}{\frac{2πm}{T}} = \frac{T}{m}.
```
We'll refer to each of those samples as ``p_j = p(t_j)``, where ``t_j``
is the time at sample ``j`` and ``j`` is an index from ``0`` to ``n-1``.

What is the discrete Fourier transform of our signal ``p_j``? We should be able
to figure that out if we can express our signal ``p(t)`` as something that
looks like FFTW's definition of the DFT. How can we do that? Well, first we
need to remember that
```math
\cos(α+β) = \cos(α)\cos(β) - \sin(α)\sin(β)
```
which lets us rewrite our signal as
```math
p(t) = A \cos(ωt+φ) = A\left[ \cos(ωt)\cos(φ) - \sin(ωt)\sin(φ) \right] = A \cos(φ)\cos(ωt) - A\sin(φ)\sin(ωt).
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
p(t) = \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{\imath ωt} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-\imath ωt}
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
p(t_j) = p_j = \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath jm/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath jm/n}
```

That's very close to the DFT definition. So what happens if we evaluate the DFT
of that last expression?
We'll call the discrete Fourier transform of the signal ``\hat{p}_k``. So...
```math
\begin{aligned}
  \hat{p}_k &= \sum_{j=0}^{n-1} p_j e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath jm/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath jm/n} \right) e^{-2 \pi \imath jk/n} \\
            &= \sum_{j=0}^{n-1} \left( \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(m+k)/n} \right) \\
            &=\frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] \sum_{j=0}^{n-1} e^{-2π\imath j(m+k)/n} 
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
```math
  \sum_{j=0}^{n-1} e^{2π\imath j(m-m)/n} = \sum_{j=0}^{n-1} e^{0} = \sum_{j=0}^{n-1} 1 = n
```
and the second sum will be 
```math
  \sum_{j=0}^{n-1} e^{-2π\imath j(m+m)/n} = \sum_{j=0}^{n-1} e^{-4π\imath jm/n} = 0
```
from the previous discussion, since ``m+k=2m \ne 0``.

For ``k = -m``, the first sum will be zero, since ``m - -m = 2m \ne 0``, and
the second sum will be
```math
  \sum_{j=0}^{n-1} e^{2π\imath j(m-m))/n} = n
```
again.

Great! So now we finally can write down the DFT of our example signal
```math
p(t) = A \cos(ωt+φ) = A \cos\left(\left[\frac{2πm}{T}\right]t+φ\right),
```
which is (wish I could figure out how to do the `cases` LaTeX environment)...
```math
\begin{aligned}
  \hat{p}_m & = \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```

### Some Special Cases
There are two special cases that we need to consider: the mean component and the Nyquist component of the DFT.
Let's try the mean component first.

#### The Mean Component
So imagine if we start out with the same signal
```math
p(t) = A \cos(ωt+φ)
```
but say that ``ω = 0``.
Since ``ω = 2πm/T``, that implies that ``m = 0`` also.
But anyway, let's rewrite that in terms of powers of ``e``:
```math
\begin{aligned}
p(t) &= A \cos(ωt+φ) \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{\imath ωt} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-\imath ωt} \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{0} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{0}\\
     &= \frac{A}{2}\left[2\cos(φ)\right]\\
     &= A\cos(φ).
\end{aligned}
```
What happens if we plug that into the definition of the discrete Fourier transform?
We'll, it's not hard to see that we'll get
```math
\begin{aligned}
\hat{p}_0 &= A\cos(φ)n \\
\hat{p}_k &= 0 \,\text{otherwise}.
\end{aligned}
```
So the two takeaways are:

  * the mean component doesn't contain a ``\frac{1}{2}`` factor
  * the mean component is always real

Next, the Nyquist frequency component.

#### Nyquist Component
The Nyquist component is the one with two samples per period, which corresponds to ``k=n/2`` in the DFT definition.
It is the highest frequency component that the DFT can resolve.
So that means our signal will look like
```math
\begin{aligned}
p(t) &= A \cos(ωt+φ) \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath jm/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath jm/n} \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(\frac{n}{2})/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(\frac{n}{2})/n}
\end{aligned}
```
Now here's where things get a bit tricky.
We've got a ``\frac{n}{2}/n`` term that we'd love to replace with ``\frac{1}{2}``.
That works fine if ``n`` is even, but what if ``n`` is odd?
We'll have to look at both cases.
First, the even case.

#### Nyquist Component with Even-Length Input
If ``n`` is even, then we can do
```math
\begin{aligned}
p(t) &= A \cos(ωt+φ) \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(\frac{n}{2})/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(\frac{n}{2})/n} \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{π\imath j} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-π\imath j}
\end{aligned}
```
and then realize that ``e^{π\imath j} = e^{-π\imath j} = \left(-1\right)^j``, which let's us simplify a bit further to
```math
p(t) = \frac{A}{2}\left[2\cos(φ)]e^{π\imath j}\right] = A\cos(φ)e^{π\imath j}
```

The next step is to think about which components of the DFT we need to worry about.
If we shove that into the DFT, we'll need to just focus on the ``e^{-2π\imath j (n/2)/n} = e^{-π\imath j}`` component of the DFT, and we'll eventually end up with
```math
\begin{aligned}
\hat{p}_{n/2} &= A\cos(φ)n \\
\hat{p}_k &= 0 \,\text{otherwise},
\end{aligned}
```
and so the takeaways are identical to the mean component:

  * the Nyquist component for ``n`` even doesn't contain a ``\frac{1}{2}`` factor
  * the Nyquist component for ``n`` even is always real

#### Nyquist Component with Odd-Length Input
So, we're starting with 
```math
\begin{aligned}
p(t) &= A \cos(ωt+φ) \\
     &= \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath j(\frac{n}{2})/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath j(\frac{n}{2})/n}
\end{aligned}
```
Now, the trick here is that there *is* no Nyquist frequency component with an odd-input DFT.
We'll never end up with ``e^{±2π\imath j(\frac{n}{2})/n} = e^{±π\imath}`` components in the DFT definition, since ``n`` is odd.
For example, if ``n=9``, the ``k`` in the original definition of the DFT
```math
  y_k = \sum_{j=0}^{n-1} x_j e^{-2 \pi \imath jk/n}
```
will never take on the value ``n/2``, since that would be ``4\frac{1}{2}`` and ``k`` is an integer.
So this special case isn't special for odd input lengths.

### Order of Outputs
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
  p(t) &= A \cos(ωt+φ) \\
  \hat{p}_m & = \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```
we would expect ``\hat{p}_m`` to appear in the ``1+m`` position (since ``m`` starts from 0, but the first "position" is 1 for Julia arrays), and ``\hat{p}_{-m}`` to appear in the ``1+n-m`` position.
But things get a bit more complicated if we use a real-input FFT (which AcousticMetrics.jl does).
See the next section.

### Real-Input FFTs and Half-Complex Format
If we look back at the results for the DFT of our simple signal
```math
\begin{aligned}
  p(t) &= A \cos(ωt+φ) \\
  \hat{p}_m & = \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```
we can't help but notice that the negative and positive frequency results are closely related.
If we know one, we can figure out the other.
And if we want to find ``A`` and ``φ``, we just need either the positive or negative result.
For example, if ``a_r`` and ``a_i`` are the real and imaginary parts of ``\hat{p}_m``, respectively, then
```math
A = \frac{2}{n}\sqrt{a_r^2 + a_i^2}
```
and
```math
φ = \arctan(a_i/a_r).
```
For the Nyquist frequency, though, we know that
```math
\begin{aligned}
\hat{p}_{n/2} &= A\cos(φ)n \\
\hat{p}_k &= 0 \,\text{otherwise},
\end{aligned}
```
and so ``a_r = A\cos(φ)n`` and ``a_i = 0``.
We have only one non-zero component, so we'll have to define 
```math
A = a_r/n
```
and
```math
φ = 0.
```
Or, wait, maybe it would be better to make `A = abs(a_r)/n` and `φ = π` if `a_r < 0`, and `φ = 0` otherwise.

So, for real-input FFTs, FFTW only gives you the non-negative frequencies of the DFT.
Finally, if we want to avoid complex numbers entirely, we can use the "real-to-real" transform that returns the DFT in the [halfcomplex format](https://www.fftw.org/fftw3_doc/The-Halfcomplex_002dformat-DFT.html).
This returns the frequencies in an order similar to the standard in-order manner discussed previously, but only returns the non-negative portion of the spectrum.
Specifically, the FFTW manual shows that the order of outputs will be
```math
r_0, r_1, r_2, ..., r_{n/2}, i_{(n+1)/2-1}, ..., i_2, i_1
```
where ``r_k`` and ``i_k`` are the real and imaginary parts of component ``k``, and division by 2 is rounded down.
An example makes this a bit more clear.
Let's imagine we have a signal of length 8, so ``n = 8``.
Then the output we'll get from FFTW will be 
```math
r_0, r_1, r_2, r_3, r_4, i_3, i_2, i_1
```
``i_0`` and ``i_4`` are "missing," but that doesn't bother us since we know that both of those are always zero for a real-input even-length DFT.

What if we had an odd-length input signal?
Let's try ``n=9`` this time.
Then the output will be 
```math
r_0, r_1, r_2, r_3, r_4, i_4, i_3, i_2, i_1
```
This time the ``i_4`` component isn't "missing," which is a good thing, since it's not zero.

### Time Offset
So far we've been assuming that the time ``t`` starts at 0.
What if that's not true, i.e., that ``t_j = t_0 + j\frac{T}{n}``?
Then
```math
ω t_j = \left( \frac{2πm}{T} \right) \left(t_0 + j \frac{T}{n} \right) = \frac{2πm t_0}{T} + \frac{2πmj}{n}
```
and the signal is now
```math
\begin{aligned}
p(t_j) = p_j &= A\cos(ω[t_0 + j\frac{T}{n}] + φ) \\
             &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath m t_0/T + 2π\imath jm/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath m t_0/T - 2π\imath jm/n} \\
             &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath m t_0/T} e^{2π\imath jm/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath m t_0/T} e^{-2π\imath jm/n}.
\end{aligned}
```
Next, we substitute the signal into the definition of the DFT:
```math
\begin{aligned}
  \hat{p}_k &= \sum_{j=0}^{n-1} p_j e^{-2 \pi \imath jk/n} \\
            &=\frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath m t_0/T} \sum_{j=0}^{n-1} e^{2π\imath j(m-k)/n} + \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath m t_0/T} \sum_{j=0}^{n-1} e^{-2π\imath j(m+k)/n} 
\end{aligned}
```
then use the same arguments we used before for the summations to find that
```math
\begin{aligned}
  \hat{p}_m & = \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath m t_0/T} n \\
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath m t_0/T} n \\
  \hat{p}_{k} & = 0\,\text{otherwise}.
\end{aligned}
```
Let's work on the non-zero ``\hat{p}`` components a bit.
First, the positive-``m`` one:
```math
\begin{aligned}
  \hat{p}_m & = \frac{A}{2}\left[\cos(φ) + \imath \sin(φ) \right] e^{2π\imath m t_0/T} n \\
            & = \frac{A}{2}\left[e^{\imath φ} \right] e^{2π\imath m t_0/T} n \\
            & = \frac{A}{2}\left[e^{\imath φ + 2π\imath m t_0/T} \right] n \\
            & = \frac{A}{2}\left[e^{\imath (φ + 2π m t_0/T)} \right] n \\
            & = \frac{A}{2}\left[\cos(φ + 2π m t_0/T) + \imath \sin(φ+ 2π m t_0/T) \right] n
\end{aligned}
```
Then, the negative-``m`` one:
```math
\begin{aligned}
  \hat{p}_{-m} & = \frac{A}{2}\left[\cos(φ) - \imath \sin(φ) \right] e^{-2π\imath m t_0/T} n \\
               & = \frac{A}{2}\left[e^{-\imath φ} \right] e^{-2π\imath m t_0/T} n \\
               & = \frac{A}{2}\left[e^{-\imath φ - 2π\imath m t_0/T} \right] n \\
               & = \frac{A}{2}\left[e^{-\imath (φ + 2π m t_0/T)} \right] n \\
               & = \frac{A}{2}\left[\cos(φ + 2π m t_0/T) - \imath \sin(φ+ 2π m t_0/T) \right] n
\end{aligned}
```
So now, if we want to find ``A`` and ``φ`` from the ``\hat{p}_m`` components ``a_r`` 
```math
a_r = \frac{A}{2}\cos(φ + 2π m t_0/T)n
```
and ``a_i``
```math
a_i = \frac{A}{2}\sin(φ + 2π m t_0/T)n
```
we can use the same formula for ``A``
```math
A = \frac{2}{n}\sqrt{a_r^2 + a_i^2}
```
and a slightly modified version of the formula for ``φ``
```math
φ = \arctan(a_i/a_r) - 2π m t_0/T.
```

And we don't need to worry about the two special cases discussed previously, since the ``φ`` angle for both the mean and Nyquist components is always zero.
What about the special cases we discussed previously (the mean and the Nyquist frequencies)?
Obviously shifting the mean component shouldn't change anything.
But what about the Nyquist frequency?

#### Nyquist Component with a Time Offset
We know from previous work that the odd-input length Nyquist component isn't special, so we'll ignore that case.
So, for the even-input length Nyquist component, we'll start with the same signal
```math
\begin{aligned}
p(t_j) = p_j &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath m t_0/T + 2π\imath jm/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath m t_0/T - 2π\imath jm/n} \\
             &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath m t_0/T} e^{2π\imath jm/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath m t_0/T} e^{-2π\imath jm/n},
\end{aligned}
```
and then say that ``m = n/2``, like previously
```math
\begin{aligned}
  p_j &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{2π\imath (n/2) t_0/T} e^{2π\imath j(n/2)/n} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-2π\imath (n/2) t_0/T} e^{-2π\imath j(n/2)/n} \\
      &= \frac{A}{2}\left[ \cos(φ) + \imath \sin(φ)\right] e^{π\imath n t_0/T} e^{π\imath j} + \frac{A}{2}\left[ \cos(φ) - \imath \sin(φ)\right] e^{-π\imath n t_0/T} e^{-π\imath j} \\
      &= \frac{A}{2}\left[e^{\imath φ}\right] e^{π\imath n t_0/T} e^{π\imath j} + \frac{A}{2}\left[ e^{-\imath φ}\right] e^{-π\imath n t_0/T} e^{-π\imath j} \\
      &= \frac{A}{2}\left[e^{\imath (φ + πn t_0/T)}\right] e^{π\imath j} + \frac{A}{2}\left[ e^{-\imath (φ + πn t_0/T)}\right] e^{-π\imath j} \\
      &= \frac{A}{2}\left[\cos(φ + πn t_0/T) + \imath \sin(φ + πn t_0/T)\right] e^{π\imath j} + \frac{A}{2}\left[ \cos(φ + πn t_0/T) - \imath \sin(φ + πn t_0/T)\right] e^{-π\imath j} \\
      &= A\cos(φ + πn t_0/T)e^{π\imath j} \\
      &= A\cos(φ + 2π (n/2) t_0/T)e^{2π\imath j (n/2)/n}
\end{aligned}
```
So this means the Fourier transform is
```math
\hat{p}_{n/2} = A\cos(φ + 2π (n/2) t_0/T) n
```
and so we can find
```math
A = a_r/n
```
and
```math
φ = -2π(n/2) t_0/T.
```

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
