using DataFrames
using CSV
using LinearAlgebra
using Distributions
using Random
using BenchmarkTools
using Plots
using DataFrames
using Statistics

#import the csv data
returns_df = CSV.read("/Users/sunyutong/Desktop/Fintech 545 Quant Risk Analysis/Project2/DailyReturn.csv", DataFrame)


# Function to calculate exponentially weighted covariance matrix
function calculate_ewma_covariance(returns_df, λ)
    n = size(returns_df, 2) - 1  # Number of assets

    # Initialize arrays
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    cumulative_w = Vector{Float64}(undef, n)
    
    # Calculate weights
    populateWeights!(x, w, cumulative_w, λ)

    # Calculate the mean returns
    mean_returns = mean(skipmissing.(eachcol(returns_df[:, 2:end])))

    # Calculate the covariance matrix using EWMA
    cov_matrix_ewma = zeros(Float64, n, n)
    for i in 1:n
        for j in 1:n
            # Handle missing values
            valid_indices = .!isnan.(returns_df[!, i+1]) .& .!isnan.(returns_df[!, j+1])
            valid_returns_i = returns_df[!, i+1][valid_indices]
            valid_returns_j = returns_df[!, j+1][valid_indices]
            
            cov_matrix_ewma[i, j] = sum((valid_returns_i .- mean_returns[i]) .* (valid_returns_j .- mean_returns[j]) .* w[valid_indices]) / sum(w[valid_indices])
        end
    end

    return mean_returns, cov_matrix_ewma
end

# Function to populate weights
function populateWeights!(x, w, cw, λ)
    n = length(x)
    tw = 0.0
    for i in 1:n
        x[i] = i
        w[i] = (1 - λ) * λ^(i-1)
        tw += w[i]
        cw[i] = tw
    end
    for i in 1:n
        w[i] = w[i] / tw
        cw[i] = cw[i] / tw
    end
end

# Calculate mean returns and covariance matrix for λ=0.94 (for example)
λ = 0.94
mean_returns_ewma, cov_matrix_ewma = calculate_ewma_covariance(returns_df, λ)

println("Mean Returns using EWMA:")
println(mean_returns_ewma)

println("\nExponentially Weighted Covariance Matrix using EWMA:")
println(cov_matrix_ewma)





# Calculate the covariance matrix from the returns data
covariance_matrix = cov(Matrix(returns_df[:, 2:end]))

# Define the simulate_pca function
function simulate_pca(cov_matrix, nsim; nval=nothing)
    # Eigenvalue decomposition
    vals, vecs = eigen(cov_matrix)

    # Julia returns values lowest to highest, flip them and the vectors
    flip = [i for i in size(vals, 1):-1:1]
    vals = vals[flip]
    vecs = vecs[:, flip]

    tv = sum(vals)

    posv = findall(x -> x >= 1e-8, vals)
    if nval !== nothing
        if nval < size(posv, 1)
            posv = posv[1:nval]
        end
    end
    vals = vals[posv]

    vecs = vecs[:, posv]

    println("Simulating with $(size(posv, 1)) PC Factors: $(sum(vals) / tv * 100)% total variance explained")
    B = vecs * diagm(sqrt.(vals))

    m = size(vals, 1)
    r = randn(m, nsim)

    (B * r)'
end

# Use the simulate_pca function with the covariance matrix
sim = simulate_pca(covariance_matrix, 10000)

# Calculate the covariance matrix of the simulated data
sim_cov = cov(sim)

println("Covariance Matrix of Simulated Data:")
println(sim_cov)



# Function to calculate cumulative variance explained by each eigenvalue for a given λ
function calculate_cumulative_variance(λ, returns_df)
    # Calculate the covariance matrix
    cov_matrix = cov(Matrix(returns_df[:, 2:end]))

    # Eigenvalue decomposition
    vals, _ = eigen(cov_matrix)

    # Sort eigenvalues in descending order
    vals_sorted = sort(vals, rev=true)

    # Calculate the cumulative variance explained by each eigenvalue
    cumulative_variance = cumsum(vals_sorted) / sum(vals_sorted)

    return cumulative_variance
end

# Values of λ to be considered
λ_values = collect(range(0.01, stop=1, step=0.01))

# Store cumulative variance explained for each λ
cumulative_variances = []

# Calculate cumulative variance for each λ
for λ in λ_values
    cumulative_variance = calculate_cumulative_variance(λ, returns_df)
    push!(cumulative_variances, cumulative_variance)
end

# Plot the cumulative variance explained for each λ
plot(λ_values, hcat(cumulative_variances...), label=map(string, 1:length(λ_values)), legend=:bottomright, xlabel="λ", ylabel="Cumulative Variance Explained", title="Cumulative Variance Explained for Different λ")


#Look at Exponential Weights
weights = DataFrame()
cumulative_weights = DataFrame()
n=100
x = Vector{Float64}(undef,n)
w = Vector{Float64}(undef,n)
cumulative_w = Vector{Float64}(undef,n)

function populateWeights!(x,w,cw, λ)
    n = size(x,1)
    tw = 0.0
    for i in 1:n
        x[i] = i
        w[i] = (1-λ)*λ^i
        tw += w[i]
        cw[i] = tw
    end
    for i in 1:n
        w[i] = w[i]/tw
        cw[i] = cw[i]/tw
    end
end

#calculated weights λ=75%
populateWeights!(x,w,cumulative_w,0.75)
weights[!,:x] = copy(x)
weights[!,Symbol("λ=0.75")] = copy(w)
cumulative_weights[!,:x] = copy(x)
cumulative_weights[!,Symbol("λ=0.75")] = copy(cumulative_w)

#calculated weights λ=90%
populateWeights!(x,w,cumulative_w,0.90)
weights[!,Symbol("λ=0.90")] = copy(w)
cumulative_weights[!,Symbol("λ=0.90")] = copy(cumulative_w)

#calculated weights λ=97%
populateWeights!(x,w,cumulative_w,0.97)
weights[!,Symbol("λ=0.97")] = copy(w)
cumulative_weights[!,Symbol("λ=0.97")] = copy(cumulative_w)

#calculated weights λ=99%
populateWeights!(x,w,cumulative_w,0.99)
weights[!,Symbol("λ=0.99")] = copy(w)
cumulative_weights[!,Symbol("λ=0.99")] = copy(cumulative_w)



cnames = names(weights)
cnames = cnames[findall(x->x!="x",cnames)]

#plot Weights
plot(weights.x,Array(weights[:,cnames]), label=hcat(cnames...))

#plot the cumulative weights
plot(cumulative_weights.x,Array(cumulative_weights[:,cnames]), label=hcat(cnames...), legend=:bottomright)



