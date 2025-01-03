```@meta
CurrentModule = AMDocs
```
# Software Quality Assurance

## Tests
AcousticMetrics.jl uses the usual Julia testing framework to implement and run tests.
The tests can be run locally after installing AcousticMetrics.jl, and are also run automatically on GitHub Actions.

To run the tests locally, from the Julia REPL, type `]` to enter the Pkg prompt, then

```julia-repl
(docs) pkg> test AcousticMetrics
     Testing AcousticMetrics
Test Summary:      | Pass  Total  Time
Fourier transforms |   16     16  9.0s
Test Summary:     | Pass  Total  Time
Pressure Spectrum |  108    108  1.7s
Test Summary:                  | Pass  Total  Time
Mean-squared Pressure Spectrum |   88     88  8.0s
Test Summary:          | Pass  Total  Time
Power Spectral Density |   88     88  0.9s
Test Summary:              | Pass  Total  Time
Proportional Band Spectrum | 1066   1066  5.3s
Test Summary: | Pass  Total  Time
OASPL         |   16     16  0.3s
Test Summary: | Pass  Total  Time
A-weighting   |    8      8  0.5s
     Testing AcousticMetrics tests passed 

(docs) pkg> 
```

(The output associated with installing all the dependencies the tests need isn't shown above.)

## Signed Commits
The AcousticMetrics.jl GitHub repository requires all commits to the `main` branch to be signed.
See the [GitHub docs on signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification) for more information.

## Reporting Bugs
Users can use the [GitHub Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues) feature to report bugs and submit feature requests.

## Some Simple Tests

### Mean-Squared Pressure Spectrum and A-Weighting
Let's start off with a simple function that's a sum of four sinusoids.

```@example msp_a_weighting
using AcousticMetrics: PressureTimeHistory, frequency, MSPSpectrumAmplitude, OASPL

omega1 = 2*pi*50.0  # 50 Hz in rad/s
omega2 = 2*pi*100.0  # 100 Hz in rad/s
omega3 = 2*pi*150.0  # 150 Hz in rad/s
omega4 = 2*pi*200.0  # 200 Hz in rad/s
```

We'll sample the simple function 64 times, and set the time period to be one cycle of the lowest non-zero frequency: 

```@example msp_a_weighting
N = 64
period = 2*pi/min(omega1, omega2, omega3, omega4)
```

The starting time shouldn't matter, so set it to some random value, then define the time levels.

```@example msp_a_weighting
t0 = 1.23
dt = period/N
t = t0 .+ (0:(N-1)).*dt
nothing # hide
```

We need to set the pressure amplitudes, and the phase offsets.
The phase offsets shouldn't affect the mean-squared pressure **amplitude** spectrum, of course.

```@example msp_a_weighting
A0 = 1.2
A1 = 2.345
A2 = 2.789
A3 = 1.12
A4 = 1.34

phi1 = 5.1
phi2 = 6.2
phi3 = 7.1
phi4 = 8.2
nothing # hide
```

Now we can create the pressure time history:

```@example msp_a_weighting
p1 = @. (A0 +
         A1*cos(omega1*(t - phi1)) +
         A2*cos(omega2*(t - phi2)) +
         A3*cos(omega3*(t - phi3)) +
         A4*cos(omega4*(t - phi4)) )
nothing # hide
```

Let's plot that.

```@example msp_a_weighting
using GLMakie
fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel="time, sec.", ylabel="pressure, Pa")
lines!(ax1, t, p1)
save("msp_a_weighting-pressure_time_history.png", fig)
nothing # hide
```
![](msp_a_weighting-pressure_time_history.png)

Looks good.
Now we can create a mean-squared pressure spectrum object:

```@example msp_a_weighting
apth = PressureTimeHistory(p1, dt, t[1])
nothing # hide
```

and then use that to find the mean-squared pressure amplitude:

```@example msp_a_weighting
msp = MSPSpectrumAmplitude(apth)
nothing # hide
```

We know what the mean-squared pressure amplitudes should be for each frequency.
The frequencies are just these (we're converting from `rad/sec` to `cycles/sec`:

```@example msp_a_weighting
freqs_expected = [0.0, omega1, omega2, omega3, omega4] ./ (2*pi)
nothing # hide
```

And, since the **root**-mean square of a sinusoid with amplitude `A` is `A/sqrt(2)`, the mean-square of each of our sinusoid components are just

```@example msp_a_weighting
msp_expected = [A0^2, A1^2/2, A2^2/2, A3^2/2, A4^2/2]
nothing # hide
```

(The zero-frequency mean-squared pressure is different, since it isn't really a sinusoid.)

Now we can find the sound pressure level using the usual formula:

```@example msp_a_weighting
pref2 = (20e-6)^2  # usual squared reference pressure in Pa
spl_msp = 10 .* log10.(msp./pref2)
spl_msp_expected = 10 .* log10.(msp_expected./pref2)
nothing # hide
```

And also calculate the overall sound pressure level (OASPL):

```@example msp_a_weighting
oaspl_apth = OASPL(apth)
oaspl_msp = OASPL(msp)
oaspl_expected = 10.0 .* log10.(sum(msp_expected[2:end])./pref2)
(oaspl_apth, oaspl_msp, oaspl_expected)
```

The OASPL calculated from both the pressure time history and the mean-squared pressure spectrum are essentially identical to the expected value, so that's good.

We can plot the SPL now, too:

```@example msp_a_weighting
using ColorSchemes: colorschemes
colors = colorschemes[:tab10]

fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel="time, sec.", ylabel="pressure, Pa")
ax2 = fig[2, 1] = Axis(fig, xlabel="frequency, Hz", ylabel="sound pressure level re: 20 μPa")

lines!(ax1, t, p1)
scatter!(ax2, freqs_expected, spl_msp_expected; label="expected", marker=:circle, strokewidth=1, color=(colors[1], 0), strokecolor=(colors[1], 1.0))
scatter!(ax2, frequency(msp), spl_msp; label="AM.jl", marker=:x, color=colors[1])

# scatter!(ax2, freqs_expected[2:end], 10.0 .* log10.(msp_Aweight_expected./pref^2); label="expected, A-weighted, OASPL = $(oaspl_Aweight_expected) dB", marker=:circle, strokewidth=1, color=(colors[2], 0), strokecolor=(colors[2], 1.0))
# scatter!(ax2, frequency(msp), 10.0 .* log10.(msp_Aweight./pref^2); label="AM.jl, A-weighted, OASPL = $(oaspl_msp_Aweight) dB", marker=:x, color=colors[2])

# axislegend(ax1; merge=true, unique=true, position=:rt)
axislegend(ax2; merge=true, unique=true, position=:rt)

ylims!(ax2, 0.0, 110.0)

save("msp_a_weighting-spl.png", fig)
nothing # hide
```
![](msp_a_weighting-spl.png)

Looks good.

Now, let's A-weight the spectrum.
There are two routines we can use to do that: [`a_weight!`](@ref), which applies the A-weighting in-place, and [`a_weight`](@ref), which returns a new spectrum object.

