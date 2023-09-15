import Pkg; Pkg.add("StatsBase")
using Distributions
using HypothesisTests
using DataFrames
using Plots
using BenchmarkTools
using StatsBase

#########################################################################################
# Test the kurtosis function for bias in small sample sizes
d = Normal(0,1)
sample_size = 100
samples = 1000
kurts = Vector{Float64}(undef,samples)
Threads.@threads for i in 1:samples
    kurts[i] = kurtosis(rand(d,sample_size))
end

#summary statistics
describe(kurts)

t = mean(kurts)/sqrt(var(kurts)/samples)
p = 2*(1 - cdf(TDist(samples-1),abs(t)))

println("p-value $p")

#using the Included TTest
ttest = OneSampleTTest(kurts,kurtosis(d))
p2 = pvalue(ttest)

println("Match the stats package test?: $(p ≈ p2)") 

#########################################################################################
# Test the skewness function for bias in small sample sizes
d = Normal(0,1)
sample_size = 10
samples = 1000
skew = Vector{Float64}(undef,samples)
Threads.@threads for i in 1:samples
    skew[i] = skewness(rand(d,sample_size))
end

#summary statistics
describe(skew)

t = mean(skew)/sqrt(var(skew)/samples)
p = 2*(1 - cdf(TDist(samples-1),abs(t)))

println("p-value $p")

#using the Included TTest
ttest = OneSampleTTest(skew,skewness(d))
p2 = pvalue(ttest)

println("Match the stats package test?: $(p ≈ p2)") 


