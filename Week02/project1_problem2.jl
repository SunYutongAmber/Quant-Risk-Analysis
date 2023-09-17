import Pkg; 
Pkg.add("DataFrames")
Pkg.add("JuMP")
Pkg.add("CSV")
Pkg.add("Ipopt")
Pkg.add("StateSpaceModels")
Pkg.add("Printf")
Pkg.add("PlotThemes")
Pkg.add("Plots")
Pkg.add("GLM")
Pkg.add("Gadfly")
Pkg.add("Winston")
Pkg.add("Plots")
Pkg.add("HypothesisTests")
Pkg.add("StatsBase")

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
using Plots
using HypothesisTests

df = CSV.read("problem2_data.csv", DataFrame)
x = df.x
y = df.y

#################################################################
#OLS for Regression

mod = lm(@formula(y ~ x), df)
error = y - (x*0.605205 .+ 0.119836)  
mean(error) 

# Perform the Hypothesis test to check normality of the error vector
# 'Agostino-Pearson's K2 test for assessing normality of data using skewness and kurtosis.
x = error
alpha = 0.05
n = length(x)
s1=sum(x)
s2 = sum(x.^2)
s3 = sum(x.^3)
s4 = sum(x.^4)
ss = s2-(s1^2/n)
v = ss/(n-1)
k3 = ((n*s3)-(3*s1*s2)+((2*(s1^3))/n))/((n-1)*(n-2))
g1 = k3/sqrt(v^3)
k4 = ((n+1)*((n*s4)-(4*s1*s3)+(6*(s1^2)*(s2/n))-((3*(s1^4))/(n^2)))/((n-1)*(n-2)*(n-3)))-((3*(ss^2))/((n-2)*(n-3)))
g2 = k4/v^2
eg1 = ((n-2)*g1)/sqrt(n*(n-1))  #measure of skewness
eg2 = ((n-2)*(n-3)*g2)/((n+1)*(n-1))+((3*(n-1))/(n+1))  #measure of kurtosis
A = eg1*sqrt(((n+1)*(n+3))/(6*(n-2)))
B = (3*((n^2)+(27*n)-70)*((n+1)*(n+3)))/((n-2)*(n+5)*(n+7)*(n+9))
C = sqrt(2*(B-1))-1
D = sqrt(C)
E = 1/sqrt(log(D))
F = A/sqrt(2/(C-1))
Zg1 = E*log(F+sqrt(F^2+1))

G = (24*n*(n-2)*(n-3))/((n+1)^2*(n+3)*(n+5))
H = ((n-2)*(n-3)*abs(g2))/((n+1)*(n-1)*sqrt(G))
J = ((6*(n^2-(5*n)+2))/((n+7)*(n+9)))*sqrt((6*(n+3)*(n+5))/((n*(n-2)*(n-3))))
K = 6+((8/J)*((2/J)+sqrt(1+(4/J^2))))
L = (1-(2/K))/(1+H*sqrt(2/(K-4)))
Zg2 = (1-(2/(9*K))-L^(1/3))/sqrt(2/(9*K))

K2 = Zg1^2 + Zg2^2  #D'Agostino-Pearson statistic
X2 = K2  #approximation to chi-distribution

df = 2.  #degrees of freedom
P=1-ccdf(Chisq(df), X2)
#println("P = $P ")
if P>alpha
    println("The error vector is normally distributed")
else
    println("The error vector is not normally distributed")
end

#################################################################
#Maximum Likelihood Estimation #MLE for Regression
df = CSV.read("problem2_data.csv", DataFrame)
x = df.x
y = df.y

function myll(s, b...)
    n = size(y,1)
    beta = collect(b)
    e = y - x.*beta
    s2 = s*s
    ll = -n/2 * log(s2 * 2 * π) - e'*e/(2*s2)
    return ll
end

#MLE Optimization problem
    mle = Model(Ipopt.Optimizer)
    set_silent(mle)

    @variable(mle, beta[i=1],start=0)
    @variable(mle, σ >= 0.0, start = 1.0)

    register(mle,:ll,2,myll;autodiff=true)
    @NLobjective(
        mle,
        Max,
        ll(σ,beta...)
##########################
    )
optimize!(mle)

println("Betas: ", value.(beta))

b_hat = inv(x'*x)*x'*y
println("OLS: ", b_hat)


error = y - (x*0.6051912114646786 .+ 0.1196)  
mean(error) 

# Perform the Hypothesis test to check normality of the error vector
# 'Agostino-Pearson's K2 test for assessing normality of data using skewness and kurtosis.
x = error
alpha = 0.05
n = length(x)
s1=sum(x)
s2 = sum(x.^2)
s3 = sum(x.^3)
s4 = sum(x.^4)
ss = s2-(s1^2/n)
v = ss/(n-1)
k3 = ((n*s3)-(3*s1*s2)+((2*(s1^3))/n))/((n-1)*(n-2))
g1 = k3/sqrt(v^3)
k4 = ((n+1)*((n*s4)-(4*s1*s3)+(6*(s1^2)*(s2/n))-((3*(s1^4))/(n^2)))/((n-1)*(n-2)*(n-3)))-((3*(ss^2))/((n-2)*(n-3)))
g2 = k4/v^2
eg1 = ((n-2)*g1)/sqrt(n*(n-1))  #measure of skewness
eg2 = ((n-2)*(n-3)*g2)/((n+1)*(n-1))+((3*(n-1))/(n+1))  #measure of kurtosis
A = eg1*sqrt(((n+1)*(n+3))/(6*(n-2)))
B = (3*((n^2)+(27*n)-70)*((n+1)*(n+3)))/((n-2)*(n+5)*(n+7)*(n+9))
C = sqrt(2*(B-1))-1
D = sqrt(C)
E = 1/sqrt(log(D))
F = A/sqrt(2/(C-1))
Zg1 = E*log(F+sqrt(F^2+1))

G = (24*n*(n-2)*(n-3))/((n+1)^2*(n+3)*(n+5))
H = ((n-2)*(n-3)*abs(g2))/((n+1)*(n-1)*sqrt(G))
J = ((6*(n^2-(5*n)+2))/((n+7)*(n+9)))*sqrt((6*(n+3)*(n+5))/((n*(n-2)*(n-3))))
K = 6+((8/J)*((2/J)+sqrt(1+(4/J^2))))
L = (1-(2/K))/(1+H*sqrt(2/(K-4)))
Zg2 = (1-(2/(9*K))-L^(1/3))/sqrt(2/(9*K))

K2 = Zg1^2 + Zg2^2  #D'Agostino-Pearson statistic
X2 = K2  #approximation to chi-distribution

df = 2.  #degrees of freedom
P=1-ccdf(Chisq(df), X2)
#println("P = $P ")
if P>alpha
    println("The error vector is normally distributed")
else
    println("The error vector is not normally distributed")
end