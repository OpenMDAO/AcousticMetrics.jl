using AcousticMetrics
# using ANOPP2
using ForwardDiff
using Random
using Test

@testset "Fourier transforms" begin

    @testset "FFTW compared to a function with a known Fourier transform" begin
        fr(t) = 2*cos(1*2*pi*t) + 4*cos(2*2*pi*t) + 6*cos(3*2*pi*t) + 8*cos(4*2*pi*t)
        fi(t) = 2*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t)
        f(t) = fr(t) + fi(t)
        @testset "imaginary only" begin
            # a*sin(2*pi*k*t) = a*(0 - 0.5*i)*exp(2*pi*i*k*t) + a*(0 + 0.5*i)*exp(-2*pi*i*k*t)
            for T in [1.0, 2.0]
                for n in [19, 20]
                    dt = T/n
                    t = (0:n-1).*dt
                    p = fi.(t)
                    freq = rfftfreq(n, dt)
                    p_fft = rfft(p)./n
                    p_fft_expected = zeros(n)
                    p_fft_expected[findlast(x->x≈1.0, freq)] = -0.5*2
                    p_fft_expected[findlast(x->x≈2.0, freq)] = -0.5*4
                    p_fft_expected[findlast(x->x≈3.0, freq)] = -0.5*6
                    p_fft_expected[findlast(x->x≈4.0, freq)] = -0.5*8
                    @test all(isapprox.(p_fft, p_fft_expected, atol=1e-12))
                end
            end
        end
        @testset "real only" begin
            # a*cos(2*pi*k*t) = a*(0.5 + 0.0*i)*exp(2*pi*i*k*t) + a*(0.5 + 0.0*i)*exp(-2*pi*i*k*t)
            for T in [1.0, 2.0]
                for n in [19, 20]
                    dt = T/n
                    t = (0:n-1).*dt
                    p = fr.(t)
                    freq = rfftfreq(n, dt)
                    p_fft = rfft(p)./n
                    p_fft_expected = zeros(n)
                    p_fft_expected[findfirst(x->x≈1.0, freq)] = 0.5*2
                    p_fft_expected[findfirst(x->x≈2.0, freq)] = 0.5*4
                    p_fft_expected[findfirst(x->x≈3.0, freq)] = 0.5*6
                    p_fft_expected[findfirst(x->x≈4.0, freq)] = 0.5*8
                    @test all(isapprox.(p_fft, p_fft_expected, atol=1e-12))
                end
            end
        end
        @testset "complex" begin
            for T in [1.0, 2.0]
                for n in [19, 20]
                    dt = T/n
                    t = (0:n-1).*dt
                    p = f.(t)
                    freq = rfftfreq(n, dt)
                    p_fft = rfft(p)./n
                    p_fft_expected = zeros(n)
                    p_fft_expected[findfirst(x->x≈1.0, freq)] = 0.5*2
                    p_fft_expected[findfirst(x->x≈2.0, freq)] = 0.5*4
                    p_fft_expected[findfirst(x->x≈3.0, freq)] = 0.5*6
                    p_fft_expected[findfirst(x->x≈4.0, freq)] = 0.5*8
                    p_fft_expected[findlast(x->x≈1.0, freq)] = -0.5*2
                    p_fft_expected[findlast(x->x≈2.0, freq)] = -0.5*4
                    p_fft_expected[findlast(x->x≈3.0, freq)] = -0.5*6
                    p_fft_expected[findlast(x->x≈4.0, freq)] = -0.5*8
                    @test all(isapprox.(p_fft, p_fft_expected, atol=1e-12))
                end
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
            y_dft = AcousticMetrics.dft_r2hc(x)
            @test all(y_dft .≈ y_fft)
        end
    end

    # Now check derivatives.
    @testset "FFTW derivatives" begin
        @testset "basic" begin
            for n in [64, 65]
                x = rand(n)
                y = similar(x)
                dy_dx_fft = ForwardDiff.jacobian(rfft!, y, x)
                dy_dx_dft = ForwardDiff.jacobian(AcousticMetrics.dft_r2hc, x)
                @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))
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
                y = AcousticMetrics.dft_r2hc(x)
                return y
            end
            dy_dx_fft = ForwardDiff.jacobian(f1_fft, [1.1, 3.5])
            dy_dx_dft = ForwardDiff.jacobian(f1_dft, [1.1, 3.5])
            @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))
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
                y = AcousticMetrics.dft_r2hc(x)
                return y
            end
            t = rand(nt)
            dy_dx_fft = ForwardDiff.jacobian(f2_fft, t)
            dy_dx_dft = ForwardDiff.jacobian(f2_dft, t)
            @test all(isapprox.(dy_dx_fft, dy_dx_dft, atol=1e-13))
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
#            # freq_half = rfftfreq(n, dt)
#            # freq_fft = zeros(n)
#            # freq_fft[1] = freq_half[1]
#            # freq_fft[2:floor(Int, n/2)+1] = freq_half[2:end]
#            # freq_fft[end:-1:floor(Int, n/2)+2] = freq_half[2:end]
#            freq_fft = rfftfreq(n, dt)
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

@testset "Narrowband Spectrum" begin
    fr(t) = 2*cos(1*2*pi*t) + 4*cos(2*2*pi*t) + 6*cos(3*2*pi*t) + 8*cos(4*2*pi*t)
    fi(t) = 2*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t)
    f(t) = fr(t) + fi(t)
    @testset "imaginary only" begin
        for T in [1.0, 2.0]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = fi.(t)
                p_fft = rfft(p)./n
                freq, nbs = nbs_from_apth(p, dt)
                nbs_expected = zeros(floor(Int, n/2)+1)
                nbs_expected[findfirst(x->x≈1, freq)] = 2*(-0.5*2)^2
                nbs_expected[findfirst(x->x≈2, freq)] = 2*(-0.5*4)^2
                nbs_expected[findfirst(x->x≈3, freq)] = 2*(-0.5*6)^2
                nbs_expected[findfirst(x->x≈4, freq)] = 2*(-0.5*8)^2
                @test all(isapprox.(nbs, nbs_expected, atol=1e-12))
            end
        end
    end
    @testset "real only" begin
        for T in [1.0, 2.0]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = fr.(t)
                p_fft = rfft(p)./n
                freq, nbs = nbs_from_apth(p, dt)
                nbs_expected = zeros(floor(Int, n/2)+1)
                nbs_expected[findfirst(x->x≈1, freq)] = 2*(0.5*2)^2
                nbs_expected[findfirst(x->x≈2, freq)] = 2*(0.5*4)^2
                nbs_expected[findfirst(x->x≈3, freq)] = 2*(0.5*6)^2
                nbs_expected[findfirst(x->x≈4, freq)] = 2*(0.5*8)^2
                @test all(isapprox.(nbs, nbs_expected, atol=1e-12))
            end
        end
    end
    @testset "complex" begin
        for T in [1.0, 2.0]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                p_fft = rfft(p)./n
                freq, nbs = nbs_from_apth(p, dt)
                nbs_expected = zeros(floor(Int, n/2)+1)
                nbs_expected[findfirst(x->x≈1, freq)] = 2*((-0.5*2)^2 + (0.5*2)^2)
                nbs_expected[findfirst(x->x≈2, freq)] = 2*((-0.5*4)^2 + (0.5*4)^2)
                nbs_expected[findfirst(x->x≈3, freq)] = 2*((-0.5*6)^2 + (0.5*6)^2)
                nbs_expected[findfirst(x->x≈4, freq)] = 2*((-0.5*8)^2 + (0.5*8)^2)
                @test all(isapprox.(nbs, nbs_expected, atol=1e-12))
            end
        end
    end
    @testset "ANOPP2 comparison" begin
        freq_a2 = Dict(
            (1, 19)=>[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            (1, 20)=>[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            (2, 19)=>[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            (2, 20)=>[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        nbs_msp_a2 = Dict(
            (1, 19)=>[1.5582437633995294e-31, 3.999999999999993, 15.999999999999993, 36.00000000000002, 64.00000000000004, 2.970402173980353e-30, 1.3245071988896e-30, 9.836413756459529e-30, 6.116106771343153e-30, 3.8956094084988235e-30],
            (1, 20)=>[1.3331749298235103e-30, 4.0, 15.999999999999986, 36.00000000000001, 64.00000000000003, 1.1146604590772899e-29, 5.5693579908603444e-30, 5.7586846081133874e-30, 6.504158163547243e-30, 1.97215226305253e-30, 1.5461673742331835e-30],
            (2, 19)=>[1.6773033259467747e-29, 1.9088486101644234e-29, 3.9999999999999907, 1.0673969779286777e-29, 15.99999999999996, 1.9295440351470734e-30, 36.0, 9.739023521247059e-32, 64.00000000000006, 1.1591872746164311e-29],
            (2, 20)=>[1.0223637331664315e-29, 8.072775188576871e-30, 3.9999999999999907, 1.1393194782051846e-29, 15.999999999999979, 7.699282434957076e-30, 36.0, 7.618353033774544e-30, 64.00000000000006, 1.52932848240695e-29, 2.4738677987730935e-29])
        for T in [1, 2]
            for n in [19, 20]
                dt = T/n
                t = (0:n-1).*dt
                p = f.(t)
                p_fft = rfft(p)./n
                freq, nbs = nbs_from_apth(p, dt)

                t_a2 = range(0, T, length=n) |> collect # This needs to be an array, since we'll eventually be passing it to C/Fortran via ccall.
                if mod(n, 2) == 0
                    p_a2 = p
                else
                    p_a2 = f.(t_a2)
                end
                # freq_a2, nbs_msp_a2, nbs_phase_a2 = ANOPP2.a2_aa_nbs(ANOPP2.a2_aa_pa, ANOPP2.a2_aa_pa, t_a2, p_a2)
                @test all(isapprox.(freq, freq_a2[(T,n)], atol=1e-12))
                @test all(isapprox.(nbs, nbs_msp_a2[(T,n)], atol=1e-12))
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
                freq, nbs = nbs_from_apth(p, dt)
                oaspl_time_domain = AcousticMetrics.oaspl_from_apth(p)
                oaspl_freq_domain = AcousticMetrics.oaspl_from_nbs(nbs)
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
                freq, nbs = nbs_from_apth(p, dt)
                oaspl_time_domain = oaspl_from_apth(p)
                oaspl_freq_domain = oaspl_from_nbs(nbs)
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
                freq, nbs = nbs_from_apth(p, dt)
                oaspl = AcousticMetrics.oaspl_from_nbs(nbs)
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
                freq, nbs = nbs_from_apth(p, dt)
                nbs_A = @. W_A(freq)*nbs
                # nbs_A_a2 = copy(nbs)
                # ANOPP2.a2_aa_weight(ANOPP2.a2_aa_a_weight, ANOPP2.a2_aa_nbs_enum, ANOPP2.a2_aa_msp, freq, nbs_A_a2)
                # Wish I could get this to match more closely. But the weighting
                # function looks pretty nasty numerically (frequencies raised to the
                # 4th power, and one of the coefficients is about 2.24e16).
                @test all(isapprox.(nbs_A, nbs_A_a2[(T_ms, n)], atol=1e-6))
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
                freq, nbs = nbs_from_apth(p, dt)
                nbs_A = @. W_A(freq)*nbs
                # This is lame. Should be able to get this to match better,
                # right?
                @test all(isapprox.(nbs_A, nbs, atol=1e-5))
            end
        end
    end
end
