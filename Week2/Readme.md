This is for week03 project 2 documents.
For problem1, implementing Exponential_Weights_cov.jl could generate the covariance matrix with exponential weights, and it also includes code on PCA simulation and weights accumulation, and accumulative variance corresponding to the eigenvalue.

pd_matrix.jl is for Problem2. In this problem, we use two methods to Compare the results of both using the Frobenius Norm. Compare the run time between the two.
And cover how does the run time of each function compares as N increases.

combined_simulation.jl is for problem3. We implement a multivariate normal simulation that allows for simulation directly from a covariance
matrix or using PCA with an optional parameter for % variance explained. And correlation matrix and variance vector 2 ways are generated in the code. In addition, we simulate 25,000 draws from each covariance matrix using:
1. Direct Simulation
2. PCA with 100% explained.
3. PCA with 75% explained.
4. PCA with 50% explained.
