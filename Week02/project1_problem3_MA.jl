
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


#MA1
#y_t = 1.0 + .05*e_t-1 + e, e ~ N(0,.01)
n = 1000
burn_in = 50
y = Vector{Float64}(undef,n)

yt_last = 1.0
d = Normal(0,0.1)
e = rand(d,n+burn_in)

for i in 2:(n+burn_in)
    global yt_last
    y_t = 1.0 + 0.5*e[i-1] + e[i]
    if i > burn_in
        y[i-burn_in] = y_t
    end
end

println(@sprintf("Mean and Var of Y: %.2f, %.4f",mean(y),var(y)))
println(@sprintf("Expected values Y: %.2f, %.4f",1.0,(1+.5^2)*.01))

plot_ts(y,imgName="ma1_acf_pacf.png",title="MA 1")

ma1 = SARIMA(y,order=(0,0,1),include_mean=true)

StateSpaceModels.fit!(ma1)
print_results(ma1)

# Function to generate MA1 process
function generate_ma1(n, theta, sigma)
    ma1_series = zeros(n)
    ma1_series[1] = randn() * sigma
    for i in 2:n
        ma1_series[i] = randn() * sigma + theta * ma1_series[i-1]
    end
    return ma1_series
end

n = 1000  # Number of time points
theta1 = 0.6  # Moving average coefficient 1
sigma = 0.1  # Standard deviation of the error

# Generate MA1 process
ma1_series = generate_ma1(n, theta1, sigma)

# Calculate and plot ACF and PACF for MA1
acf_ma1 = autocor(ma1_series, 1:20)
pacf_ma1 = pacf(ma1_series, 1:20)

plot(1:20, acf_ma1, label="ACF", xlabel="Lag", ylabel="Autocorrelation", title="MA1 ACF and PACF")
plot!(1:20, pacf_ma1, label="PACF")

#MA2
# Function to generate MA2 process
function generate_ma2(n, theta1, theta2, sigma)
    ma2_series = zeros(n)
    ma2_series[1:2] .= randn(2) * sigma
    for i in 3:n
        ma2_series[i] = randn() * sigma + theta1 * ma2_series[i-1] + theta2 * ma2_series[i-2]
    end
    return ma2_series
end

theta2 = 0.3  # Moving average coefficient 2

# Generate MA2 process
ma2_series = generate_ma2(n, theta1, theta2, sigma)

# Calculate and plot ACF and PACF for MA2
acf_ma2 = autocor(ma2_series, 1:20)
pacf_ma2 = pacf(ma2_series, 1:20)

plot(1:20, acf_ma2, label="ACF", xlabel="Lag", ylabel="Autocorrelation", title="MA2 ACF and PACF")
plot!(1:20, pacf_ma2, label="PACF")

#MA3
# Function to generate MA3 process
function generate_ma3(n, theta1, theta2, theta3, sigma)
    ma3_series = zeros(n)
    ma3_series[1:3] .= randn(3) * sigma
    for i in 4:n
        ma3_series[i] = randn() * sigma + theta1 * ma3_series[i-1] + theta2 * ma3_series[i-2] + theta3 * ma3_series[i-3]
    end
    return ma3_series
end

theta3 = -0.2  # Moving average coefficient 3

# Generate MA3 process
ma3_series = generate_ma3(n, theta1, theta2, theta3, sigma)

# Calculate and plot ACF and PACF for MA3
acf_ma3 = autocor(ma3_series, 1:20)
pacf_ma3 = pacf(ma3_series, 1:20)

plot(1:20, acf_ma3, label="ACF", xlabel="Lag", ylabel="Autocorrelation", title="MA3 ACF and PACF")
plot!(1:20, pacf_ma3, label="PACF")
