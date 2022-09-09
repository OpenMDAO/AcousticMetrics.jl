using AcousticMetrics: p_ref
using AcousticMetrics: r2rfftfreq, rfft, rfft!, irfft, irfft!, RFFTCache, dft_r2hc, dft_hc2r
using AcousticMetrics: PressureTimeHistory, PressureSpectrum, NarrowbandSpectrum, PowerSpectralDensity
using AcousticMetrics: starttime, timestep, pressure, frequency, amplitude, halfcomplex, phase, OASPL
using AcousticMetrics: W_A
using ForwardDiff
using JLD2
using Random
using Test

include(joinpath(@__DIR__, "gen_anopp2_data", "test_functions.jl"))


@testset "Fourier transforms" begin

    @testset "FFTW compared to a function with a known Fourier transform" begin
        for T in [1.0, 2.0]
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                p_fft = rfft(p)./n
                p_fft_expected = similar(p_fft)
                p_fft_expected[1] = 6.0
                p_fft_expected[2] = 0.5*8*cos(0.2)
                p_fft_expected[end] = 0.5*8*sin(0.2)
                p_fft_expected[3] = 0.5*2.5*cos(-3.0)
                p_fft_expected[end-1] = 0.5*2.5*sin(-3.0)
                p_fft_expected[4] = 0.5*9*cos(3.1)
                p_fft_expected[end-2] = 0.5*9*sin(3.1)
                p_fft_expected[5] = 0.5*0.5*cos(-1.1)
                p_fft_expected[end-3] = 0.5*0.5*sin(-1.1)
                if n == 10
                    p_fft_expected[6] = 3*cos(0.2)
                else
                    p_fft_expected[6] = 0.5*3*cos(0.2)
                    p_fft_expected[end-4] = 0.5*3*sin(0.2)
                end
                @test all(isapprox.(p_fft, p_fft_expected, atol=1e-12))
            end
        end
    end

    @testset "FFTW vs ad-hoc discrete Fourier transform" begin
        # Should check both even- and odd-length inputs, since the definition of the
        # discrete Fourier transform output depends slightly on that.
        for n in [64, 65]
            x = rand(n)
            y_fft = similar(x)
            rfft!(y_fft, x)
            y_dft = dft_r2hc(x)
            @test all(y_dft .≈ y_fft)

            y_ifft = similar(x)
            irfft!(y_ifft, x)
            y_idft = dft_hc2r(x)
            @test all(y_idft .≈ y_ifft)
        end
    end

    # Now check derivatives.
    @testset "FFTW derivatives" begin
        @testset "basic" begin
            for n in [64, 65]
                x = rand(n)
                y = similar(x)
                dy_dx_fft = ForwardDiff.jacobian(rfft!, y, x)
                dy_dx_dft = ForwardDiff.jacobian(dft_r2hc, x)
                @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))

                dy_dx_ifft = ForwardDiff.jacobian(irfft!, y, x)
                dy_dx_idft = ForwardDiff.jacobian(dft_hc2r, x)
                @test all(isapprox.(dy_dx_ifft, dy_dx_idft, atol=1e-13))
            end
        end

        @testset "as intermediate function" begin
            function f1_fft(t)
                x = range(t[begin], t[end], length=8)
                x = @. 2*x^2 + 3*x + 5
                y = similar(x)
                rfft!(y, x)
                return y
            end
            function f1_dft(t)
                x = range(t[begin], t[end], length=8)
                x = @. 2*x^2 + 3*x + 5
                y = dft_r2hc(x)
                return y
            end
            dy_dx_fft = ForwardDiff.jacobian(f1_fft, [1.1, 3.5])
            dy_dx_dft = ForwardDiff.jacobian(f1_dft, [1.1, 3.5])
            @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))

            function f1_ifft(t)
                x = range(t[begin], t[end], length=8)
                x = @. 2*x^2 + 3*x + 5
                y = similar(x)
                irfft!(y, x)
                return y
            end
            function f1_idft(t)
                x = range(t[begin], t[end], length=8)
                x = @. 2*x^2 + 3*x + 5
                y = dft_hc2r(x)
                return y
            end
            dy_dx_ifft = ForwardDiff.jacobian(f1_ifft, [1.1, 3.5])
            dy_dx_idft = ForwardDiff.jacobian(f1_idft, [1.1, 3.5])
            @test all(isapprox.(dy_dx_ifft, dy_dx_idft, atol=1e-13))
        end

        @testset "with user-supplied cache" begin
            nx = 8
            nt = 5
            cache = RFFTCache(Float64, nx, nt)
            function f2_fft(t)
                xstart = sum(t)
                xend = xstart + 2
                x = range(xstart, xend, length=nx)
                x = @. 2*x^2 + 3*x + 5
                y = similar(x)
                rfft!(y, x, cache)
                return y
            end
            function f2_dft(t)
                xstart = sum(t)
                xend = xstart + 2
                x = range(xstart, xend, length=nx)
                x = @. 2*x^2 + 3*x + 5
                y = dft_r2hc(x)
                return y
            end
            t = rand(nt)
            dy_dx_fft = ForwardDiff.jacobian(f2_fft, t)
            dy_dx_dft = ForwardDiff.jacobian(f2_dft, t)
            @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))

            function f2_ifft(t)
                xstart = sum(t)
                xend = xstart + 2
                x = range(xstart, xend, length=nx)
                x = @. 2*x^2 + 3*x + 5
                y = similar(x)
                irfft!(y, x, cache)
                return y
            end
            function f2_idft(t)
                xstart = sum(t)
                xend = xstart + 2
                x = range(xstart, xend, length=nx)
                x = @. 2*x^2 + 3*x + 5
                y = dft_hc2r(x)
                return y
            end
            t = rand(nt)
            dy_dx_ifft = ForwardDiff.jacobian(f2_ifft, t)
            dy_dx_idft = ForwardDiff.jacobian(f2_idft, t)
            @test all(isapprox.(dy_dx_ifft, dy_dx_idft, atol=1e-13))
        end
    end
end

#@testset "Narrowband Spectrum" begin
#    f(t) = 2.0*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t) #+ 10*sin(5*2*pi*t) #+ 12*sin(6*2*pi*t)
#    # So, what do we expect the narrow-band spectrum is for this? Well, first we
#    # need to know what the Fourier transform is. Well,
#    #
#    #   a*sin(k*2*pi*t) = a*(0 - 0.5*i)*exp(i*k*2*pi*t) + a*(0 + 0.5*i)*exp(-i*k*2*pi*t)
#    #
#    # Then I would expect, for the function above, only imaginary parts of the
#    # spectrum would be non-zero, right? Right. And the R2HC only keeps the
#    # positive frequencies, so I'd expect -0.5*a, but in reverse order. So
#    # that'd be [0.0, 0.0, 0.0, ..., -5, -4, -3, -2, -1]. Great, that's exactly
#    # what I see. Wait, are these amplitudes at the right frequencies? Yep,
#    # looks good. Great. So, now, what do I expect the nbs to be? Well,
#    # according to the ANOPP2 docs, the NBS is the squared amplitude of the FFT
#    # components. Great. But after finding that, I need to multiply the non-zero
#    # parts by 2 to get the other half of the spectrum. And since the real parts
#    # of my test function are all zero, I'd expect the answer to be
#    #
#    # [0.0, 2*(-1)^2, 2*(-2)^2, 2*(-3)^2, 2*(-4)^2, 2*(-5)^2, 0.0, 0.0, ...]
#    # [0.0, 2, 8, 18, 32, 50, 0.0, 0.0, ...]
#    #
#    # Nice, that's what I'm getting, with both ANOPP2 and my stuff. But when the
#    # input length is even, I have to evaluate the pressure on a different time
#    # grid then what I pass in to ANOPP2, which is really strange. But, really,
#    # the NBS calculation doesn't depend on the time grid—the frequency and NBS
#    # are actually two entirely seperate calculations. So it seems that, when
#    # passing an *odd*-length input to `a2_aa_nbs`, ANOPP2 ignores the last
#    # entry of the pressure input, and passes the rest to the FFT. OK, I guess
#    # that's OK. But at least I know what the answer is.
#    for T in [1.0, 2.0]
#        for n in [19, 20]
#            dt = T/n
#            t = (0:n-1).*dt
#            # t = collect(t)
#            p = f.(t)
#            p_fft = rfft(p)./n
#            # freq_half = r2rfftfreq(n, dt)
#            # freq_fft = zeros(n)
#            # freq_fft[1] = freq_half[1]
#            # freq_fft[2:floor(Int, n/2)+1] = freq_half[2:end]
#            # freq_fft[end:-1:floor(Int, n/2)+2] = freq_half[2:end]
#            freq_fft = r2rfftfreq(n, dt)
#            # p = rand(n)
#            # freq, nbs = nbs_from_apth(t, p)
#            freq, nbs = nbs_from_apth(p, dt)
#            freq = collect(freq)
#            t_a2 = range(0, T, length=n) |> collect # This needs to be an array, since we'll eventually be passing it to C/Fortran via ccall.
#            if mod(n, 2) == 0
#                p_a2 = p
#            else
#                p_a2 = f.(t_a2)
#            end
#            # t_a2 = t |> collect
#            freq_a2, nbs_msp_a2, nbs_phase_a2 = ANOPP2.a2_aa_nbs(ANOPP2.a2_aa_pa, ANOPP2.a2_aa_pa, t_a2, p_a2)
#            @show T n
#            @show freq_fft p_fft
#            @show freq nbs
#            @show freq_a2 nbs_msp_a2
#            # @show freq freq_a2
#            # @show nbs .- nbs_msp_a2
#            # @show nbs nbs_msp_a2
#        end
#    end

# end

@testset "Pressure Spectrum" begin
    @testset "t0 == 0" begin
        for T in [1.0, 2.0]
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                ps = PressureSpectrum(ap)
                amp = amplitude(ps)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                phase_expected = similar(phase(ps))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = 3*cos(0.2)
                    phase_expected[6] = 0
                else
                    amp_expected[6] = 3
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps).*amplitude(ps), phase_expected.*amp_expected; atol=1e-12))

                # Make sure I can go from a PressureSpectrum to an PressureTimeHistory.
                ap_from_ps = PressureTimeHistory(ps)
                @test timestep(ap_from_ps) ≈ timestep(ap)
                @test starttime(ap_from_ps) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_ps), pressure(ap)))
            end
        end
        @testset "negative amplitudes" begin
            for T in [1.0, 2.0]
                f(t) = -6 - 8*cos(1*2*pi/T*t + 0.2) - 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
                for n in [10, 11]
                    dt = T/n
                    t = (0:n-1).*dt
                    p = f.(t)
                    ap = PressureTimeHistory(p, dt)
                    ps = PressureSpectrum(ap)
                    freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                    amp_expected = similar(amplitude(ps))
                    amp_expected[1] = 6
                    amp_expected[2] = 8
                    amp_expected[3] = 2.5
                    amp_expected[4] = 9
                    amp_expected[5] = 0.5
                    phase_expected = similar(phase(ps))
                    phase_expected[1] = pi
                    phase_expected[2] = -pi + 0.2
                    phase_expected[3] = pi - 3
                    phase_expected[4] = 3.1
                    phase_expected[5] = -1.1
                    # Handle the Nyquist frequency (kinda tricky). There isn't really a
                    # Nyquist frequency for the odd input length case.
                    if n == 10
                        amp_expected[6] = 3*cos(0.2)
                        phase_expected[6] = 0
                    else
                        amp_expected[6] = 3
                        phase_expected[6] = 0.2
                    end
                    @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                    @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                    @test all(isapprox.(phase(ps).*amplitude(ps), phase_expected.*amp_expected; atol=1e-12))

                    # Make sure I can go from a PressureSpectrum to an PressureTimeHistory.
                    ap_from_ps = PressureTimeHistory(ps)
                    @test timestep(ap_from_ps) ≈ timestep(ap)
                    @test starttime(ap_from_ps) ≈ starttime(ap)
                    @test all(isapprox.(pressure(ap_from_ps), pressure(ap)))
                end
            end
        end
    end

    @testset "t0 !== 0" begin
        for T in [1.0, 2.0]
            t0 = 0.13
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                t = t0 .+ (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt, t0)
                ps = PressureSpectrum(ap)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                phase_expected = similar(phase(ps))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = abs(3*cos(5*2*pi/T*t0 + 0.2))
                    phase_expected[6] = rem2pi(pi - 5*2*pi/T*t0, RoundNearest)
                else
                    amp_expected[6] = 3
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps), phase_expected; atol=1e-12))

                # Make sure I can go from a PressureSpectrum to an PressureTimeHistory.
                ap_from_ps = PressureTimeHistory(ps)
                @test timestep(ap_from_ps) ≈ timestep(ap)
                @test starttime(ap_from_ps) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_ps), pressure(ap)))
            end
        end
    end
end

@testset "Narrowband Spectrum" begin
    @testset "t0 == 0" begin
        for T in [1.0, 2.0]
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(nbs))
                amp_expected[1] = 6^2
                amp_expected[2] = 0.5*8^2
                amp_expected[3] = 0.5*2.5^2
                amp_expected[4] = 0.5*9^2
                amp_expected[5] = 0.5*0.5^2
                phase_expected = similar(phase(nbs))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = (3*cos(0.2))^2
                    phase_expected[6] = 0
                else
                    amp_expected[6] = 0.5*3^2
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(nbs), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(nbs), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(nbs).*amplitude(nbs), phase_expected.*amp_expected; atol=1e-12))

                # Make sure I can convert a NBS to a pressure spectrum.
                ps = PressureSpectrum(nbs)
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = 3*cos(0.2)
                else
                    amp_expected[6] = 3
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps).*amplitude(ps), phase_expected.*amp_expected; atol=1e-12))

                # Make sure I can convert a NBS to the acoustic pressure.
                ap_from_nbs = PressureTimeHistory(nbs)
                @test timestep(ap_from_nbs) ≈ timestep(ap)
                @test starttime(ap_from_nbs) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_nbs), pressure(ap)))
            end
        end
    end

    @testset "t0 !== 0" begin
        for T in [1.0, 2.0]
            t0 = 0.13
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                t = t0 .+ (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt, t0)
                nbs = NarrowbandSpectrum(ap)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(nbs))
                amp_expected[1] = 6^2
                amp_expected[2] = 0.5*8^2
                amp_expected[3] = 0.5*2.5^2
                amp_expected[4] = 0.5*9^2
                amp_expected[5] = 0.5*0.5^2
                phase_expected = similar(phase(nbs))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    # amp_expected[6] = (3*cos(5*2*pi/T*t0 + 0.2))^2
                    # phase_expected[6] = rem2pi(-5*2*pi/T*t0, RoundNearest)
                    amp_expected[6] = (3*cos(5*2*pi/T*t0 + 0.2))^2
                    phase_expected[6] = rem2pi(pi - 5*2*pi/T*t0, RoundNearest)
                else
                    amp_expected[6] = 0.5*3^2
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(nbs), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(nbs), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(nbs), phase_expected; atol=1e-12))

                # Make sure I can convert a NBS to a pressure spectrum.
                ps = PressureSpectrum(nbs)
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    # The `t0` term pushes the cosine below zero, which messes
                    # up the test. Hmm... what's the right thing to do here?
                    # Well, what should the phase and amplitude be?
                    # amp_expected[6] = 3*cos(5*2*pi/T*t0 + 0.2)
                    amp_expected[6] = abs(3*cos(5*2*pi/T*t0 + 0.2))
                else
                    amp_expected[6] = 3
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps), phase_expected; atol=1e-12))

                # Make sure I can convert a NBS to the acoustic pressure.
                ap_from_nbs = PressureTimeHistory(nbs)
                @test timestep(ap_from_nbs) ≈ timestep(ap)
                @test starttime(ap_from_nbs) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_nbs), pressure(ap)))
            end
        end
    end

    @testset "ANOPP2 comparison" begin
        a2_data = load(joinpath(@__DIR__, "gen_anopp2_data", "nbs.jld2"))
        freq_a2 = a2_data["a2_nbs_freq"]
        nbs_msp_a2 = a2_data["a2_nbs_amp"]
        nbs_phase_a2 = a2_data["a2_nbs_phase"]
        for T in [1, 2]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = apth_for_nbs.(t)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                @test all(isapprox.(frequency(nbs), freq_a2[(T,n)], atol=1e-12))
                @test all(isapprox.(amplitude(nbs), nbs_msp_a2[(T,n)], atol=1e-12))
                # Checking the phase is tricky, since it involves the ratio of
                # the imaginary component to the real component of the MSP
                # spectrum (definition is phase = atan(imag(fft(p)),
                # real(fft(p)))). For the components of the spectrum that have
                # zero amplitude that ratio ends up being very noisy. So scale
                # the phase by the amplitude to remove the problematic
                # zero-amplitude components.
                @test all(isapprox.(phase(nbs).*amplitude(nbs), nbs_phase_a2[T, n].*nbs_msp_a2[T, n], atol=1e-12))
            end
        end
    end
end

@testset "Power Spectral Density" begin
    @testset "t0 == 0" begin
        for T in [1.0, 2.0]
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                df = 1/T
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                psd = PowerSpectralDensity(ap)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(psd))
                amp_expected[1] = 6^2/df
                amp_expected[2] = 0.5*8^2/df
                amp_expected[3] = 0.5*2.5^2/df
                amp_expected[4] = 0.5*9^2/df
                amp_expected[5] = 0.5*0.5^2/df
                phase_expected = similar(phase(psd))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = (3*cos(0.2))^2/df
                    phase_expected[6] = 0
                else
                    amp_expected[6] = 0.5*3^2/df
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(psd), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(psd), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(psd).*amplitude(psd), phase_expected.*amp_expected; atol=1e-12))

                # Make sure I can convert a PSD to a pressure spectrum.
                ps = PressureSpectrum(psd)
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    amp_expected[6] = 3*cos(0.2)
                else
                    amp_expected[6] = 3
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps).*amplitude(ps), phase_expected.*amp_expected; atol=1e-12))

                # Make sure I can convert a PSD to the acoustic pressure.
                ap_from_psd = PressureTimeHistory(psd)
                @test timestep(ap_from_psd) ≈ timestep(ap)
                @test starttime(ap_from_psd) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_psd), pressure(ap)))
            end
        end
    end

    @testset "t0 !== 0" begin
        for T in [1.0, 2.0]
            t0 = 0.13
            f(t) = 6 + 8*cos(1*2*pi/T*t + 0.2) + 2.5*cos(2*2*pi/T*t - 3.0) + 9*cos(3*2*pi/T*t + 3.1) + 0.5*cos(4*2*pi/T*t - 1.1) + 3*cos(5*2*pi/T*t + 0.2)
            for n in [10, 11]
                dt = T/n
                df = 1/T
                t = t0 .+ (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt, t0)
                psd = PowerSpectralDensity(ap)
                freq_expected = [0.0, 1/T, 2/T, 3/T, 4/T, 5/T]
                amp_expected = similar(amplitude(psd))
                amp_expected[1] = 6^2/df
                amp_expected[2] = 0.5*8^2/df
                amp_expected[3] = 0.5*2.5^2/df
                amp_expected[4] = 0.5*9^2/df
                amp_expected[5] = 0.5*0.5^2/df
                phase_expected = similar(phase(psd))
                phase_expected[1] = 0
                phase_expected[2] = 0.2
                phase_expected[3] = -3
                phase_expected[4] = 3.1
                phase_expected[5] = -1.1
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    # amp_expected[6] = (3*cos(5*2*pi/T*t0 + 0.2))^2
                    # phase_expected[6] = rem2pi(-5*2*pi/T*t0, RoundNearest)
                    amp_expected[6] = (3*cos(5*2*pi/T*t0 + 0.2))^2/df
                    phase_expected[6] = rem2pi(pi - 5*2*pi/T*t0, RoundNearest)
                else
                    amp_expected[6] = 0.5*3^2/df
                    phase_expected[6] = 0.2
                end
                @test all(isapprox.(frequency(psd), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(psd), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(psd), phase_expected; atol=1e-12))

                # Make sure I can convert a PSD to a pressure spectrum.
                ps = PressureSpectrum(psd)
                amp_expected = similar(amplitude(ps))
                amp_expected[1] = 6
                amp_expected[2] = 8
                amp_expected[3] = 2.5
                amp_expected[4] = 9
                amp_expected[5] = 0.5
                # Handle the Nyquist frequency (kinda tricky). There isn't really a
                # Nyquist frequency for the odd input length case.
                if n == 10
                    # The `t0` term pushes the cosine below zero, which messes
                    # up the test. Hmm... what's the right thing to do here?
                    # Well, what should the phase and amplitude be?
                    # amp_expected[6] = 3*cos(5*2*pi/T*t0 + 0.2)
                    amp_expected[6] = abs(3*cos(5*2*pi/T*t0 + 0.2))
                else
                    amp_expected[6] = 3
                end
                @test all(isapprox.(frequency(ps), freq_expected; atol=1e-12))
                @test all(isapprox.(amplitude(ps), amp_expected; atol=1e-12))
                @test all(isapprox.(phase(ps), phase_expected; atol=1e-12))

                # Make sure I can convert a PSD to the acoustic pressure.
                ap_from_psd = PressureTimeHistory(psd)
                @test timestep(ap_from_psd) ≈ timestep(ap)
                @test starttime(ap_from_psd) ≈ starttime(ap)
                @test all(isapprox.(pressure(ap_from_psd), pressure(ap)))
            end
        end
    end

    @testset "ANOPP2 comparison" begin
        a2_data = load(joinpath(@__DIR__, "gen_anopp2_data", "psd.jld2"))
        freq_a2 = a2_data["a2_psd_freq"]
        psd_msp_a2 = a2_data["a2_psd_amp"]
        psd_phase_a2 = a2_data["a2_psd_phase"]
        for T in [1, 2]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = apth_for_nbs.(t)
                ap = PressureTimeHistory(p, dt)
                psd = PowerSpectralDensity(ap)
                @test all(isapprox.(frequency(psd), freq_a2[(T,n)], atol=1e-12))
                @test all(isapprox.(amplitude(psd), psd_msp_a2[(T,n)], atol=1e-12))
                # Checking the phase is tricky, since it involves the ratio of
                # the imaginary component to the real component of the MSP
                # spectrum (definition is phase = atan(imag(fft(p)),
                # real(fft(p)))). For the components of the spectrum that have
                # zero amplitude that ratio ends up being very noisy. So scale
                # the phase by the amplitude to remove the problematic
                # zero-amplitude components.
                @test all(isapprox.(phase(psd).*amplitude(psd), psd_phase_a2[T, n].*psd_msp_a2[T, n], atol=1e-12))
            end
        end
    end
end

@testset "OASPL" begin
    @testset "Parseval's theorem" begin
        fr(t) = 2*cos(1*2*pi*t) + 4*cos(2*2*pi*t) + 6*cos(3*2*pi*t) + 8*cos(4*2*pi*t)
        fi(t) = 2*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t)
        f(t) = fr(t) + fi(t)
        for T in [1.0, 2.0]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                oaspl_time_domain = OASPL(ap)
                oaspl_freq_domain = OASPL(nbs)
                @test oaspl_freq_domain ≈ oaspl_time_domain
            end
        end
    end
    @testset "function with know mean squared pressure" begin
        f(t) = 4*cos(2*2*pi*t)
        # What's the mean-square of that function? I think the mean-square of
        # 
        #   f(t) = a*cos(2*pi*k*t) 
        # 
        # is a^2/2. So
        for T in [1.0, 2.0]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                msp_expected = 4^2/2
                oaspl_expected = 10*log10(msp_expected/p_ref^2)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                oaspl_time_domain = OASPL(ap)
                oaspl_freq_domain = OASPL(nbs)
                @test oaspl_time_domain ≈ oaspl_expected
                @test oaspl_freq_domain ≈ oaspl_expected
            end
        end
    end
    @testset "ANOPP2 comparison" begin
        fr(t) = 2*cos(1*2*pi*t) + 4*cos(2*2*pi*t) + 6*cos(3*2*pi*t) + 8*cos(4*2*pi*t)
        fi(t) = 2*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t)
        f(t) = fr(t) + fi(t)
        oaspl_a2 = 114.77121254719663
        for T in [1, 2]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                oaspl = OASPL(ap)
                # oaspl_a2 = ANOPP2.a2_aa_oaspl(ANOPP2.a2_aa_nbs_enum, ANOPP2.a2_aa_msp, freq, nbs)
                @test isapprox(oaspl, oaspl_a2, atol=1e-12)
            end
        end
    end
end

@testset "A-weighting" begin
    @testset "ANOPP2 comparison" begin
        fr(t) = 2*cos(1e3*2*pi*t) + 4*cos(2e3*2*pi*t) + 6*cos(3e3*2*pi*t) + 8*cos(4e3*2*pi*t)
        fi(t) = 2*sin(1e3*2*pi*t) + 4*sin(2e3*2*pi*t) + 6*sin(3e3*2*pi*t) + 8*sin(4e3*2*pi*t)
        f(t) = fr(t) + fi(t)
        nbs_A_a2 = Dict(
          (1, 19)=>[0.0, 4.000002539852234, 21.098932320239594, 47.765983983028875, 79.89329612328712, 6.904751939255882e-29, 3.438658433244509e-29, 3.385314868430938e-29, 4.3828241499153937e-29, 3.334042101984942e-29],
          (1, 20)=>[0.0, 4.000002539852235, 21.09893232023959, 47.76598398302881, 79.89329612328707, 2.4807405180395723e-29, 3.319538256490389e-29, 1.1860147288201262e-29, 1.5894684286161776e-29, 9.168407004474984e-30, 1.4222371367588704e-31],
          (2, 19)=>[0.0, 4.137956256384954e-30, 4.00000253985224, 2.1118658029791977e-29, 21.098932320239633, 3.4572972532471526e-29, 47.765983983028924, 1.2630134771692395e-28, 79.89329612328716, 8.284388048614786e-29],
          (2, 20)=>[0.0, 1.2697180778261437e-30, 4.000002539852251, 4.666290179209354e-29, 21.098932320239584, 3.4300386105764425e-29, 47.76598398302884, 6.100255343320017e-29, 79.89329612328727, 1.801023480958872e-28, 6.029776808298499e-29],
        )
        for T_ms in [1, 2]
            for n in [19, 20]
                dt = T_ms*1e-3/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                # nbs_A = W_A(nbs)
                # amp_A = W_A(nbs)
                amp_A = W_A.(frequency(nbs)).*amplitude(nbs)
                # ANOPP2.a2_aa_weight(ANOPP2.a2_aa_a_weight, ANOPP2.a2_aa_nbs_enum, ANOPP2.a2_aa_msp, freq, nbs_A_a2)
                # Wish I could get this to match more closely. But the weighting
                # function looks pretty nasty numerically (frequencies raised to the
                # 4th power, and one of the coefficients is about 2.24e16).
                # @show T_ms n amp_A nbs_A_a2[(T_ms, n)]
                @test all(isapprox.(amp_A, nbs_A_a2[(T_ms, n)], atol=1e-6))
            end
        end
    end

    @testset "1kHz check" begin
        # A 1kHz signal should be unaffected by A-weighting.
        fr(t) = 2*cos(1e3*2*pi*t)
        fi(t) = 2*sin(1e3*2*pi*t)
        f(t) = fr(t) + fi(t)
        for T_ms in [1, 2]
            for n in [19, 20]
                dt = T_ms*1e-3/n
                t = (0:n-1).*dt
                p = f.(t)
                ap = PressureTimeHistory(p, dt)
                nbs = NarrowbandSpectrum(ap)
                amp = amplitude(nbs)
                # nbs_A = W_A(nbs)
                # amp_A = W_A(nbs)
                amp_A = W_A.(frequency(nbs)).*amplitude(nbs)
                # This is lame. Should be able to get this to match better,
                # right?
                @test all(isapprox.(amp_A, amp, atol=1e-5))
            end
        end
    end
end
