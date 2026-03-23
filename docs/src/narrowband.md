```@meta
CurrentModule = AMDocs
```
# A Practical Narrowband Example
In the [Theory](@ref) section we spent a bunch of time understanding how a discrete Fourier transform works.
Now that we know all that, let's actually try out AcousticMetrics.jl and make sure we get the Right Answer™.

## The Acoustic Pressure Time History
First, we need to create an acoustic pressure time history, i.e. acoustic pressure as a function of time.
In [Theory](@ref) we used a simple function 
```math
p(t) = A \cos(ωt+φ)
```
But to represent this acoustic pressure on a computer, we have to, of course, sample it for a finite number of times.
Let's imagine that the frequency of our pressure time history is 1000 Hz.
Then our angular frequency ``ω`` will be
```@example narrowband1
ω = 2*pi*1000.0
```
(`ω` is in units of radians per second of course.)
Let's further assume that we'll use a sampling rate of 32,000 Hz.
That would imply a time step size `Δt` of...
```@example narrowband1
sampling_rate = 32_000.0
dt = 1/sampling_rate
```
Finally, let's assume the phase offset ``φ`` is zero, ``A`` is `0.0001` Pascals, and we'll sample our time history 128 times.
Then we can use AcousticMetrics.jl's `PressureTimeHistory` to represent the time history this way:
```@example narrowband1
num_samples = 128
t = (0:(num_samples-1)) .* dt
A = 0.0001
p = @. A * cos(ω*t)
```
And make a plot to admire our beautiful cosine:
```@example narrowband1
using GLMakie
fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel="time, sec.", ylabel="pressure, Pa")
lines!(ax1, t, p)
save("narrowband1-pressure_time_history.png", fig)
nothing # hide
```
![](narrowband1-pressure_time_history.png)

Now, we need to create a special `struct` to tell AcousticMetrics.jl 
