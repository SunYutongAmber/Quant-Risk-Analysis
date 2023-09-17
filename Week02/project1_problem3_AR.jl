
using StatsBase
using CSV
using Distributions
using StatsBase
using DataFrames
using CSV
using Plots
using PlotThemes
using Printf
using JuMP
using Ipopt
using StateSpaceModels
using LinearAlgebra
using GLM
using Gadfly
using Winston
using StatsBase: autocor
using Plots
using Pkg
Pkg.add("Plots")


#AR1
# Function to generate AR1 process
function generate_ar1(n, phi, sigma)
    ar1_series = zeros(n)
    ar1_series[1] = randn()
    for i in 2:n
        ar1_series[i] = phi * ar1_series[i-1] + sigma * rand(Normal(0,1))
    end
    return ar1_series
end

n = 1000  # Number of time points
burn_in = 50
phi = 0.5  # Autoregressive coefficient
sigma = 0.1  # Standard deviation of the error

# Generate AR1 process
ar1_series = generate_ar1(n + burn_in, phi, sigma)
ar1_series = ar1_series[(burn_in + 1):end]  # Remove burn-in period

# Calculate and plot ACF and PACF
acf_ar1 = autocor(ar1_series, 1:20)
pacf_ar1 = pacf(ar1_series, 1:20)

plot(1:20, acf_ar1, label="ACF", xlabel="Lag", ylabel="Autocorrelation", title="AR1 ACF and PACF")
plot!(1:20, pacf_ar1, label="PACF")


using Distributions  # For generating random numbers
using Plots
using StatsBase

# Function to generate AR1 process
function generate_ar1(n, phi, sigma)
    ar1_series = zeros(n)
    ar1_series[1] = randn()
    for i in 2:n
        ar1_series[i] = phi * ar1_series[i-1] + sigma * rand(Normal(0,1))
    end
    return ar1_series
end

n = 1000  # Number of time points
burn_in = 50
phi = 0.5  # Autoregressive coefficient
sigma = 0.1  # Standard deviation of the error

# Generate AR1 process
ar1_series = generate_ar1(n + burn_in, phi, sigma)
ar1_series = ar1_series[(burn_in + 1):end]  # Remove burn-in period

# Calculate and plot ACF and PACF
acf_ar1 = autocor(ar1_series, 1:20)
pacf_ar1 = pacf(ar1_series, 1:20)

plot(1:20, acf_ar1, label="ACF", xlabel="Lag", ylabel="Autocorrelation", title="AR1 ACF and PACF")
plot!(1:20, pacf_ar1, label="PACF")


#AR2


# Function to generate AR2 process
function generate_ar2(n, phi1, phi2, sigma)
    ar2_series = zeros(n)
    ar2_series[1:2] .= randn(2)  # Initial values
    for i in 3:n
        ar2_series[i] = phi1 * ar2_series[i-1] + phi2 * ar2_series[i-2] + sigma * randn()
    end
    return ar2_series
end

n = 1000  # Number of time points
burn_in = 50
phi1 = 0.5  # Autoregressive coefficient 1
phi2 = -0.3  # Autoregressive coefficient 2
sigma = 0.1  # Standard deviation of the error

# Generate AR2 process
ar2_series = generate_ar2(n + burn_in, phi1, phi2, sigma)
ar2_series = ar2_series[(burn_in + 1):end]  # Remove burn-in period

# Calculate ACF and PACF for AR2 process
lags = 40
acf_ar2 = autocor(ar2_series, 1:lags)
pacf_ar2 = acf_ar2 .- [sum(acf_ar2[1:k]) for k in 1:lags]

# Plot AR2 process, ACF, and PACF
plot!(ar2_series, xlabel="Time", ylabel="Value", label="AR2 Process", legend=:topleft)
plot!(1:lags, acf_ar2, xlabel="Lag", ylabel="ACF", label="ACF")
plot!(1:lags, pacf_ar2, xlabel="Lag", ylabel="PACF", label="PACF")




# AR3
# Function to generate AR3 process
function generate_ar3(n, phi1, phi2, phi3, sigma)
    ar3_series = zeros(n)
    ar3_series[1:3] .= randn(3)  # Initial values
    for i in 4:n
        ar3_series[i] = phi1 * ar3_series[i-1] + phi2 * ar3_series[i-2] + phi3 * ar3_series[i-3] + sigma * randn()
    end
    return ar3_series
end

n = 1000  # Number of time points
burn_in = 50
phi1 = 0.4  # Autoregressive coefficient 1
phi2 = -0.2  # Autoregressive coefficient 2
phi3 = 0.1  # Autoregressive coefficient 3
sigma = 0.1  # Standard deviation of the error

# Generate AR3 process
ar3_series = generate_ar3(n + burn_in, phi1, phi2, phi3, sigma)
ar3_series = ar3_series[(burn_in + 1):end]  # Remove burn-in period

# Calculate ACF and PACF for AR3 process
acf_ar3 = autocor(ar3_series, 1:lags)
pacf_ar3 = acf_ar3 .- [sum(acf_ar3[1:k]) for k in 1:lags]

# Plot AR3 process and ACF/PACF
plot(ar3_series, xlabel="Time", ylabel="Value", label="AR3 Process", legend=:topleft)
plot(1:lags, acf_ar3, xlabel="Lag", ylabel="ACF", label="ACF")
plot(1:lags, pacf_ar3, xlabel="Lag", ylabel="PACF", label="PACF")



