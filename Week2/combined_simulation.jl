using Random

# Generate a random covariance matrix
function generate_random_covariance_matrix(n)
    A = randn(n, n)
    return A' * A
end

# Generate correlation matrix from covariance matrix
function generate_correlation_matrix(cov_matrix)
    diag_inv_sqrt = inv(Diagonal(sqrt.(diag(cov_matrix)))')
    return diag_inv_sqrt * cov_matrix * diag_inv_sqrt
end

# Generate variance vector from covariance matrix
function generate_variance_vector(cov_matrix)
    return diag(cov_matrix)
end

# Generate correlation matrix using exponentially weighted correlation
function generate_ewma_correlation_matrix(cov_matrix, λ)
    n = size(cov_matrix, 1)
    correlation_matrix = zeros(Float64, n, n)

    for i in 1:n
        for j in 1:n
            correlation_matrix[i, j] = cov_matrix[i, j] / sqrt(cov_matrix[i, i] * cov_matrix[j, j])
        end
    end

    return (1 - λ) * λ.^(abs.(1:n .- 1)) .* correlation_matrix
end

# Generate variance vector using exponentially weighted variance
function generate_ewma_variance_vector(cov_matrix, λ)
    n = size(cov_matrix, 1)
    variance_vector = diag(cov_matrix)
    return (1 - λ) * λ.^(abs.(1:n .- 1)) .* variance_vector
end

# Set the matrix size
n = 5

# Generate a random covariance matrix
random_cov_matrix = generate_random_covariance_matrix(n)

# Generate the correlation matrix and variance vector using standard Pearson correlation/variance
correlation_matrix_standard = generate_correlation_matrix(random_cov_matrix)
variance_vector_standard = generate_variance_vector(random_cov_matrix)

# Generate the correlation matrix and variance vector using exponentially weighted correlation/variance with λ = 0.97
λ = 0.97
correlation_matrix_ewma = generate_ewma_correlation_matrix(random_cov_matrix, λ)
variance_vector_ewma = generate_ewma_variance_vector(random_cov_matrix, λ)

println("Standard Pearson Correlation Matrix:")
println(correlation_matrix_standard)
println("\nStandard Pearson Variance Vector:")
println(variance_vector_standard)

println("\nExponentially Weighted Correlation Matrix (λ = $λ):")
println(correlation_matrix_ewma)
println("\nExponentially Weighted Variance Vector (λ = $λ):")
println(variance_vector_ewma)



# Generate covariance matrix using Pearson correlation and variance
function generate_covariance_matrix(correlation_matrix, variance_vector)
    return diagm(variance_vector) * correlation_matrix * diagm(variance_vector)
end

# Simulate draws using direct simulation
function simulate_direct(cov_matrix, nsim)
    n = size(cov_matrix, 1)
    L = cholesky(Symmetric(cov_matrix))
    r = randn(n, nsim)
    return L.L * r
end

# Simulate draws using PCA with specified explained variance
function simulate_pca(cov_matrix, explained_variance, nsim)
    vals, vecs = eigen(cov_matrix)
    n = size(cov_matrix, 1)
    k = sum(vals .> 1e-8 * vals[1])  # Number of principal components to keep
    explained_variance_ratio = sum(vals[1:k]) / sum(vals)
    ratio = 0.0
    idx = 1
    while ratio < explained_variance
        ratio = sum(vals[1:idx]) / sum(vals)
        idx += 1
    end
    idx = max(1, idx - 1)  # Ensure at least one component is included
    k = min(k, idx)  # Ensure k doesn't exceed the available components
    B = vecs[:, 1:k]
    r = randn(k, nsim)
    return B * diagm(sqrt.(vals[1:k])) * r
end

# Calculate the covariance of the simulated values
function simulated_covariance(simulated_values)
    return cov(simulated_values')
end

# Calculate the Frobenius norm
function frobenius_norm(A, B)
    return norm(A - B, "fro")
end

# Run simulation and calculate Frobenius norm
function run_simulation_and_compare(cov_matrix, nsim, explained_variance)
    # Simulate using direct simulation
    simulated_direct = simulate_direct(cov_matrix, nsim)
    cov_direct = simulated_covariance(simulated_direct)

    # Simulate using PCA
    simulated_pca = simulate_pca(cov_matrix, explained_variance, nsim)
    cov_pca = simulated_covariance(simulated_pca)

    # Calculate Frobenius norm
    norm_direct = frobenius_norm(cov_matrix, cov_direct)
    norm_pca = frobenius_norm(cov_matrix, cov_pca)

    return norm_direct, norm_pca
end

# Set the random seed for reproducibility
Random.seed!(123)

# Set matrix size
n = 5

# Generate the covariance matrices
correlation_matrix_standard = generate_correlation_matrix(random_cov_matrix)
cov_matrix_standard = generate_covariance_matrix(correlation_matrix_standard, variance_vector_standard)

correlation_matrix_ewma = generate_ewma_correlation_matrix(random_cov_matrix, λ)
cov_matrix_ewma = generate_covariance_matrix(correlation_matrix_ewma, variance_vector_ewma)

# Run simulations and compare
nsim = 25000

# Direct Simulation
println("Direct Simulation")
@btime run_simulation_and_compare($cov_matrix_standard, $nsim, 1.0)

# PCA with 100% explained
println("\nPCA with 100% Explained")
@btime run_simulation_and_compare($cov_matrix_standard, $nsim, 1.0)

# PCA with 75% explained
println("\nPCA with 75% Explained")
@btime run_simulation_and_compare($cov_matrix_standard, $nsim, 0.75)

# PCA with 50% explained
println("\nPCA with 50% Explained")
@btime run_simulation_and_compare($cov_matrix_standard, $nsim, 0.5)
