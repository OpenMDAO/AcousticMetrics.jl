function apth_for_nbs(t)
    fr = 2*cos(1*2*pi*t) + 4*cos(2*2*pi*t) + 6*cos(3*2*pi*t) + 8*cos(4*2*pi*t)
    fi = 2*sin(1*2*pi*t) + 4*sin(2*2*pi*t) + 6*sin(3*2*pi*t) + 8*sin(4*2*pi*t)
    return fr + fi
end

function apth_for_pbs(freq0, t)
    return 6 + 8*cos(1*2*pi*freq0*t + 0.2) + 2.5*cos(2*2*pi*freq0*t - 3.0) + 9*cos(3*2*pi*freq0*t + 3.1)
end

psd_func(freq) = 3*freq/1e3 + (4e-1)*(freq/1e3)^2 + (5e-2)*(freq/1e3)^3 + (6e-3)*(freq/1e3)^4
psd_func_int(freq_l, freq_r) = (((1/2)*3*(freq_r/1e3)^2 + (1/3)*(4e-1)*(freq_r/1e3)^3 + (1/4)*(5e-2)*(freq_r/1e3)^4 + (1/5)*(6e-3)*(freq_r/1e3)^5) - ((1/2)*3*(freq_l/1e3)^2 + (1/3)*(4e-1)*(freq_l/1e3)^3 + (1/4)*(5e-2)*(freq_l/1e3)^4 + (1/5)*(6e-3)*(freq_l/1e3)^5))*1e3
