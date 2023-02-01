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
