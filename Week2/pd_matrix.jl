# Cholesky that assumes PD matrix
function chol_pd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        root[j,j] =  sqrt(a[j,j] .- s);

        ir = 1.0/root[j,j]
        #update off diagonal rows of the column
        for i in (j+1):n
            s = root[i,1:(j-1)]' * root[j,1:(j-1)]
            root[i,j] = (a[i,j] - s) * ir 
        end
    end
end


n=5
sigma = fill(0.9,(n,n))
for i in 1:n
    sigma[i,i]=1.0
end

root = Array{Float64,2}(undef,(n,n))

chol_pd!(root,sigma)

root*root' ≈ sigma

root2 = cholesky(sigma).L
root ≈ root2

#make the matrix PSD
sigma[1,2] = 1.0
sigma[2,1] = 1.0
eigvals(sigma)

chol_pd!(root,sigma)

#Cholesky that assumes PSD
function chol_psd!(root,a)
    n = size(a,1)
    #Initialize the root matrix with 0 values
    root .= 0.0

    #loop over columns
    for j in 1:n
        s = 0.0
        #if we are not on the first column, calculate the dot product of the preceeding row values.
        if j>1
            s =  root[j,1:(j-1)]'* root[j,1:(j-1)]
        end
  
        #Diagonal Element
        temp = a[j,j] .- s
        if 0 >= temp >= -1e-8
            temp = 0.0
        end
        root[j,j] =  sqrt(temp);

        #Check for the 0 eigan value.  Just set the column to 0 if we have one
        if 0.0 == root[j,j]
            root[j,(j+1):n] .= 0.0
        else
            #update off diagonal rows of the column
            ir = 1.0/root[j,j]
            for i in (j+1):n
                s = root[i,1:(j-1)]' * root[j,1:(j-1)]
                root[i,j] = (a[i,j] - s) * ir 
            end
        end
    end
end


chol_psd!(root,sigma)

root*root' ≈ sigma

root2 = cholesky(sigma).L

#make the matrix slightly non-definite
sigma[1,2] = 0.7357
sigma[2,1] = 0.7357
eigvals(sigma)

chol_psd!(root,sigma)

using LinearAlgebra
using BenchmarkTools
using Random

# Generate a non-PSD correlation matrix of size n x n
function generate_non_psd_corr_matrix(n)
    # Generate a random correlation matrix
    A = randn(n, n)
    C = A' * A

    # Make it non-PSD by adding some negative eigenvalues
    eigvals_C = eigen(C).values
    eigvals_C[end] = -abs(eigvals_C[end])  # Change the last eigenvalue to negative

    return eigvecs(C) * Diagonal(eigvals_C) * eigvecs(C)'
end

# Fix the matrix using Higham's method
function fix_psd_higham(A)
    eigvals, eigvecs = eigen(Hermitian(A))
    eigvals[eigvals .< 0] .= 0
    return eigvecs * Diagonal(eigvals) * eigvecs'
end

# Fix the matrix using near_psd() from StatsBase
function fix_psd_near_psd(A)
    using StatsBase
    return near_psd(A)
end

# Measure Frobenius norm
function frobenius_norm(A)
    return norm(A, "fro")
end

# Compare runtimes of both methods
function compare_runtimes(n)
    A = generate_non_psd_corr_matrix(n)

    time_higham = @elapsed fix_psd_higham(A)
    time_near_psd = @elapsed fix_psd_near_psd(A)

    println("Run time for Higham's method: $time_higham seconds")
    println("Run time for near_psd(): $time_near_psd seconds")
end

# Compare Frobenius norms
function compare_frobenius_norm(n)
    A = generate_non_psd_corr_matrix(n)
    A_higham = fix_psd_higham(A)
    A_near_psd = fix_psd_near_psd(A)

    norm_higham = frobenius_norm(A - A_higham)
    norm_near_psd = frobenius_norm(A - A_near_psd)

    println("Frobenius norm difference for Higham's method: $norm_higham")
    println("Frobenius norm difference for near_psd(): $norm_near_psd")
end

# Set matrix size
n = 500

# Compare runtimes
compare_runtimes(n)

# Compare Frobenius norms
compare_frobenius_norm(n)
