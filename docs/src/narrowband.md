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
But let's use something more interesting, like
```math
p(t) = A_1 \cos(ω_1t+φ_1) + A_2 \cos(ω_2t+φ_2) + A_3 \cos(ω_3t+φ_3)
```

To represent this acoustic pressure on a computer, we have to, of course, sample it for a finite number of times.
Let's imagine that the frequencies of our pressure time history will be 500 Hz, 1000 Hz, and 2000 Hz.
Then we can create an vector to hold our three angular frequencies like this
```@example narrowband1
ω = 2*pi*[500.0, 1000.0, 2000.0]
```
(`ω` is in units of radians per second of course.)
Let's further assume that we'll use a sampling rate of 32,000 Hz.
That would imply a time step size `Δt` of...
```@example narrowband1
sampling_rate = 32_000.0
dt = 1/sampling_rate
```
Finally, we'll make up some values for the amplitudes 

```@example narrowband1
A = [0.0003, 0.0002, 0.0001]
```

where `A` is in units of Pascals, and phase

```@example narrowband1
φ = [0.1, 0.2, 0.3]
```

which is in radians.
We'll sample the time history 256 times.
That means the total length of our time history will be
```@example narrowband1
num_samples = 256
t_max = num_samples * dt
```
The lowest-frequency part of our time history has a period of
```@example narrowband1
period_ω1 = 2*pi/ω[1]
```
So we can calculate how often our time history will repeat itself by dividing the total length of the history by the period of the lowest-frequency component:
```@example narrowband1
n_repeats = t_max / period_ω1
```

Anyway, we can create out pressure time history now, first by creating the time array:

```@example narrowband1
t = (0:(num_samples-1)) .* dt
```

And then using that to construct the pressure time history:

```@example narrowband1
p = @. A[1] * cos(ω[1]*t + φ[1]) + A[2] * cos(ω[2]*t + φ[2]) + A[3] * cos(ω[3]*t + φ[3])
```

Let's use Makie to make a plot to admire our beautiful time history:

```@example narrowband1
using GLMakie
fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel="time, sec.", ylabel="pressure, Pa")
lines!(ax1, t, p)
save("narrowband1-pressure_time_history.png", fig)
nothing # hide
```
![](narrowband1-pressure_time_history.png)

Indeed, it does repeat the number of times we expected it to (`n_repeats`).

Now, we need to create a special `struct` called `PressureTimeHistory` to get our pressure time history in a form that AcousticMetrics.jl can work with.
There are two things we need to create the  `PressureTimeHistory` `struct`: the pressure values (`p` in this example), and the time step (`dt` in this example).
(The `PressureTimeHistory` constructor can also accept a starting time value `t0`, but uses `0` if you don't provide it).
We create the `struct` with:

```@example narrowband1
using AcousticMetrics
apth = PressureTimeHistory(p, dt)
```

Great!
What type is `apth`?

```@example narrowband1
@show typeof(apth)
```

What can we do with our `PressureTimeHistory` `apth`?
Well, it's an `AbstractVector` that returns the pressure when indexed:

```@example narrowband1
@show p[8] apth[8]
nothing # hide
```

(Notice that `p[8]` is the same as `apth[8]`, as we'd expect.)
And so we can do the usual things that we can do with a plain Julia `Vector`:

```@example narrowband1
@show size(apth) length(apth) apth[4:2:8]
nothing # hide
```

There are other acoustic-specific methods that we can use with `PressureTimeHistory`.
For example, we can get the time step size:

```@example narrowband1
@show AcousticMetrics.timestep(apth)
nothing # hide
```

And the sample rate

```@example narrowband1
@show AcousticMetrics.samplerate(apth)
nothing # hide
```

And the starting time value

```@example narrowband1
@show AcousticMetrics.starttime(apth)
nothing # hide
```

We can also get a range of the time values for each pressure value using the `time` method:

```@example narrowband1
t = AcousticMetrics.time(apth)
```

which can be useful for plotting, etc..

## Narrowband Pressure Spectra
Now let's get the spectrum of the pressure amplitude.
We can do that by passing our pressure time history to the `PressureSpectrumAmplitude` constructor, which will create a `PressureSpectrumAmplitude` `struct` of course:

```@example narrowband1
pressure_amp = PressureSpectrumAmplitude(apth)
```

Now, a `PressureSpectrumAmplitude` is also an `AbstractVector`, so we can do the usual vector things:

```@example narrowband1
@show size(pressure_amp) length(pressure_amp)
nothing # hide
```

Indexing `pressure_amp` will give us the pressure spectrum value at a particular frequency.
The first entry in `pressure_amp` will contain the zero-frequency component, which is zero for our example:

```@example narrowband1
pressure_amp[1]
```

`pressure_amp` also knows the sample rate of the pressure time history that was used to construct it:

```@example narrowband1
AcousticMetrics.samplerate(pressure_amp)
```

We can get the bin width of the spectrum, aka the spacing between each frequency:

```@example narrowband1
AcousticMetrics.frequencystep(pressure_amp)
```

There is also a `frequency` method that will give you a vector of frequencies, one for each entry in `pressure_amp`:

```@example narrowband1
freq = AcousticMetrics.frequency(pressure_amp)
```

That's very useful for plotting.
Let's plot our pressure spectrum:

```@example narrowband1
fig2 = Figure()
ax2_1 = fig2[1, 1] = Axis(fig2, xlabel="frequency, Hz", ylabel="pressure, Pa", xticks=0:500:4000)
scatter!(ax2_1, freq, pressure_amp)
xlims!(ax2_1, 0.0, 4000)
save("narrowband1-pressure_amp_spectrum.png", fig2)
```
![](narrowband1-pressure_amp_spectrum.png)

That plot matches what we expect: we have non-zero entries for `500.0 Hz`, `1000.0 Hz`, and `2000.0 Hz`, with values that match what we defined with `A` above.

We could also get the phase spectrum, which isn't very commonly used in acoustics:

```@example narrowband1
pressure_phase = PressureSpectrumPhase(apth)
```

The `PressureSpectrumPhase` `struct` `pressure_phase` is also an `AbstractVector`, and can do the usual vector things.
The `samplerate`, `frequencystep`, `frequency`, etc. methods also work.
So we *could* create a plot to check that the phases match what we set with `φ`.
But that gets a bit tricky, since the phase of the zero components of the spectrum is pretty nonsensical.
We're only interested in the phase of the non-zero components of the pressure spectrum, so let's find the indices that correspond to those:

```@example narrowband1
idx_nonzero = findall(pressure_amp .> 1e-10)
```

And then restrict our plotting to those indices:

```@example narrowband1
freq_phase = AcousticMetrics.frequency(pressure_phase)
fig3 = Figure()
ax3_1 = fig3[1, 1] = Axis(fig3, xlabel="frequency, Hz", ylabel="phase, rad", xticks=0:500:4000)
scatter!(ax3_1, freq_phase[idx_nonzero], pressure_phase[idx_nonzero])
xlims!(ax3_1, 0.0, 4000)
save("narrowband1-pressure_phase_spectrum.png", fig3)
```
![](narrowband1-pressure_phase_spectrum.png)

That plot matches the `φ` variable we defined at the beginning, so we're happy with that.

## Narrowband Mean-Squared Spectra
The pressure spectrum is useful for testing purposes, but acoustic metrics usually deal with the spectrum of mean-squared pressure, not just plain pressure.
We can get the amplitude of the mean-squared pressure spectrum via

```@example narrowband1
msp_amp = MSPSpectrumAmplitude(apth)
```

Again, a `MSPSpectrumAmplitude` is an `AbstractVector`, and so can be used like one.
And it also works with the the `samplerate`, `frequencystep`, `frequency`, etc. methods.

Now, what do we expect the mean-squared amplitude to look like for our test case?
Well, the mean-square of a sinusoid is equal to half it's squared amplitude (see e.g. [the Wikipedia page on Root Mean Square](https://en.wikipedia.org/wiki/Root_mean_square)).
So, for our example, we would expect that to be:

```@example narrowband1
0.5 .* A.^2
```

(That is not true for the zero and Nyquist frequency, as we saw on the [Theory](@ref) page.
AcousticMetrics.jl will do the correct thing for those two cases.)

Let's do yet another plot to make sure:

```@example narrowband1
freq4 = AcousticMetrics.frequency(msp_amp)
fig4 = Figure()
ax4_1 = fig4[1, 1] = Axis(fig4, xlabel="frequency, Hz", ylabel="mean squared pressure, Pa^2", xticks=0:500:4000)
scatter!(ax4_1, freq4, msp_amp, marker='x', markersize=20, label="MSPSpectrumAmplitude")
scatter!(ax4_1, ω./(2*pi), 0.5 .* A.^2, marker='+', markersize=25, label="0.5*A^2")
xlims!(ax4_1, 0.0, 4000)
axislegend(ax4_1)
save("narrowband1-msp_amp_spectrum.png", fig4)
```
![](narrowband1-msp_amp_spectrum.png)

Perfect agreement, yay.

What about the phase of the mean-squared pressure?
We can calculate that by creating a `MSPSpectrumPhase`, but that is the same thing as the phase of the pressure spectrum.
So in AcousticMetrics.jl the `MSPSpectrumPhase` is just an alias for `PressureSpectrumPhase`.

## Narrowband Power Spectral Density
Another commonly-used acoustic metric is the power spectral density (PSD), which is the mean-squared pressure divided by the narrowband bandwidth, i.e. the spacing between frequency values, i.e. the thing returned by `AcousticMetrics.frequencystep`, i.e. the inverse of the period associated with the pressure time history:

```@example narrowband1
df = 1/t_max
@show df AcousticMetrics.frequencystep(msp_amp)
nothing # hide
```

In our example, we would expect the PSD to look like this, then:

```@example narrowband1
0.5 .* A.^2 ./ df
```

To calculate the PSD amplitude using AcousticMetrics.jl, we use the `PowerSpectralDensityAmplitude` constructor:

```@example narrowband1
psd_amp = PowerSpectralDensityAmplitude(apth)
```

which, yet again, is an `AbstractVector`, and works with all the acoustic methods we've discussed so far.

Now, let's plot it and compare to what we think we should get.

```@example narrowband1
freq5 = AcousticMetrics.frequency(psd_amp)
fig5 = Figure()
ax5_1 = fig5[1, 1] = Axis(fig5, xlabel="frequency, Hz", ylabel="power spectral density, Pa^2/Hz", xticks=0:500:4000)
scatter!(ax5_1, freq5, psd_amp, marker='x', markersize=20, label="PowerSpectralDensityAmplitude")
scatter!(ax5_1, ω./(2*pi), 0.5 .* A.^2 ./ df, marker='+', markersize=25, label="0.5*A^2/df")
xlims!(ax5_1, 0.0, 4000)
axislegend(ax5_1)
save("narrowband1-psd_amp_spectrum.png", fig5)
```
![](narrowband1-psd_amp_spectrum.png)

What about the phase of the PSD?
Like the mean-squared pressure case, AcousticMetrics.jl provides a `PowerSpectralDensityPhase`, but it's just an alias for `PressureSpectrumPhase`.

## Tonal vs Narrowband Spectra
All of the narrowband metrics we've talked about so far keep track of a boolean type parameter called `IsTonal`, which indicates whether the spectrum is considered a "tonal" or a "regular" narrowband spectrum.
The `IsTonal` parameter is decided by you, the user, when you create any of the narrowband spectra, via an optional `istonal` argument.
By default `istonal` is `false`.
If we wanted to use `true` instead, we could do something like:

```@example narrowband1
it = true
msp_amp_tonal = MSPSpectrumAmplitude(apth, it)
```

There is an `istonal` method that we can use to check that it worked:

```@example narrowband1
@show AcousticMetrics.istonal(msp_amp_tonal) AcousticMetrics.istonal(msp_amp)
nothing # hide
```

What's the difference?
We won't see any difference in the amplitudes, frequencies, phase, or anything else associated with the narrowband spectra between `msp_amp` and `msp_amp_tonal`.
The importance tonal vs non-tonal comes into play when we start working with proportional band spectra, where we will combine the acoustic energy of a range of narrowband frequencies into a proportional band.
In AcousticMetrics.jl, if a narrowband spectrum is tonal, the acoustic energy (read: mean-squared pressure) is assumed to be concentrated at the center of each narrowband frequency.
If the spectrum is non-tonal, the energy is assumed to be evenly distributed throughout each narrow frequency band.

One detail to note: AcousticMetrics.jl doesn't allow you to create a PSD amplitude from a tonal spectrum, since the PSD is not well-defined for tonal signals.
Since the PSD is defined as the mean-squared pressure divided by the frequency bin width `df`, the power spectral density of a tone will increase without limit as `df` is decreased, since a tone by definition is non-zero at discrete frequencies.

## Converting From One Narrowband Metric to Another
We can convert from one narrowband spectrum to another this way:

```@example narrowband1
psd_amp2 = PowerSpectralDensityAmplitude(msp_amp)
@show maximum(abs.(psd_amp2 .- psd_amp))
nothing # hide
```

Going from a mean-squared pressure amplitude to a power spectral density amplitude might not be suprising, but we could also convert the phase to a PSD:

```@example narrowband1
psd_amp3 = PowerSpectralDensityAmplitude(pressure_phase)
@show maximum(abs.(psd_amp3 .- psd_amp))
nothing # hide
```

How is that possible?
The answer is that all of the narrowband spectra types that we've talked about are actually just small wrappers around the raw Fourier transform of the pressure time history.
We can grab the underlying `Vector` that holds the Fourier transform (in FFTW's "half complex" format) using the `halfcomplex` method:

```@example narrowband1
@show AcousticMetrics.halfcomplex(msp_amp) AcousticMetrics.halfcomplex(psd_amp)
nothing # hide
```

We can use the strong equality operator `===` to prove to ourselves that `msp_amp` and `psd_amp2` actually just hold a reference to the same Fourier transform `Vector`:

```@example narrowband1
@show AcousticMetrics.halfcomplex(msp_amp) === AcousticMetrics.halfcomplex(psd_amp2)
nothing # hide
```

## A-Weighting
[A-weighting](https://en.wikipedia.org/wiki/A-weighting) involves increasing or decreasing an acoustic spectrum at specific frequencies to mimic how the sound is perceived by a human.
We can A-weight any narrowband spectrum defined in AcousticMetrics.jl using the `a_weight` and `a_weight!` methods.
As the names imply, `a_weight` will return a new spectrum with the appropriate weighting applied, while the `a_weight!` will weight the input spectrum in-place:

```@example narrowband1
msp_A = a_weight(msp_amp)
# Make a copy of `msp_amp` before we weight it:
msp_A2 = deepcopy(msp_amp)
a_weight!(msp_A2)
@show maximum(abs.(msp_A .- msp_A2))
nothing # hide
```

AcousticMetrics.jl also exposes a function `W_A` that takes a frequency and returns the appropriate weighting factor for the frequency.
We can use that to plot the A-weighting curve, but first we'll calculate the gain in decibels associated with the A-weighting:

```@example narrowband1
gain_A = 10 .* log10.(W_A.(freq))
extrema(gain_A)
```

And here's the plot:

```@example narrowband1
fig6 = Figure()
ax6_1 = fig6[1, 1] = Axis(fig6, xlabel="frequency, Hz", ylabel="A-weighting gain, dB", xticks=0:500:4000)
scatter!(ax6_1, freq, gain_A, marker='x', markersize=20)
xlims!(ax6_1, 0.0, 4000)
ylims!(ax6_1, -10, 2)
save("narrowband1-a_weighting.png", fig6)
```
![](narrowband1-a_weighting.png)
