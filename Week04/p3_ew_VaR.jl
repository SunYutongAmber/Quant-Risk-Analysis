using DataFrames
using CSV
using Random
using Statistics

# Read the CSV file into a DataFrame
portfolio_data = CSV.File("portfolio.csv") |> DataFrame
daily_prices = DataFrame(CSV.File("DailyPrices.csv"))
daily_returns = return_calculate(daily_prices, "DISCRETE", "Date")

function return_calculate(prices, method="DISCRETE", dateColumn="Date")
    vars_ = names(prices)
    nVars = length(vars_)
    vars_ = [var for var in vars_ if var != dateColumn]
    
    if nVars == length(vars_)
        error("dateColumn: $dateColumn not in DataFrame: $vars_")
    end
    
    nVars -= 1
    p = convert(Matrix, prices[vars_])
    n, m = size(p)
    p2 = Matrix{Float64}(undef, n - 1, m)
    
    if uppercase(method) == "DISCRETE" || uppercase(method) == "LOG"
        # Calculate returns
        for i in 1:n - 1
            for j in 1:m
                p2[i, j] = p[i + 1, j] / p[i, j]
            end
        end
        
        if uppercase(method) == "DISCRETE"
            p2 .-= 1.0
        else
            p2 .= log.(p2)
        end
    elseif uppercase(method) == "CLASSIC"
        for i in 1:n - 1
            for j in 1:m
                p2[i, j] = p[i + 1, j] - p[i, j]
            end
        end
    else
        error("method: $method must be in ('LOG', 'DISCRETE', 'CLASSIC')")
    end
    
    dates = convert(Array, prices[dateColumn][2:n])
    num = DataFrame([dateColumn => dates])
    
    for i in 1:nVars
        num[vars_[i]] = p2[:, i]
    end
    
    return num
end


function price(portfolio, prices, portfolio_name, Delta=false)
    if portfolio_name == "All"
        assets = combine(groupby(portfolio, :Stock, sort=true), "Holding" => sum)
    else
        assets = filter(row -> row.Portfolio == portfolio_name, portfolio)
    end
    
   
    current_price = dot(prices[!, assets.Stock][end, :], assets.Holding)
    holdings = assets.Holding
    stock_codes = unique(assets.Stock)
    assets_prices = hcat(prices.Date, prices[!, stock_codes])
    asset_values = assets.Holding .* prices[!, assets.Stock][end, :]
    delta = asset_values / current_price
    
    return current_price, assets_prices, holdings
end


# Example usage
#portfolio_data = CSV.File("portfolio.csv") |> DataFrame
#aily_prices_data = CSV.File("DailyPrices.csv") |> DataFrame

function calculate(returns, method="DISCRETE", date_column="Date")
    variables = names(returns)
    num_variables = length(variables)
    variables = filter(x -> x != date_column, variables)

    num_variables -= 1
    price_data = Matrix(returns[variables])
    num_rows, num_cols = size(price_data)
    returns_matrix = similar(price_data, Float64, num_rows - 1, num_cols)

    for i in 1:num_rows - 1
        for j in 1:num_cols
            returns_matrix[i, j] = price_data[i + 1, j] / price_data[i, j]
        end
    end

    if uppercase(method) == "DISCRETE"
        returns_matrix .-= 1.0
    elseif uppercase(method) == "LOG"
        returns_matrix .= log.(returns_matrix)
    else
        error("method: $method must be in ('LOG', 'DISCRETE')")
    end

    dates = Array(returns[!, date_column][2:num_rows])
    num = DataFrame([Symbol(date_column) => dates])

    for i in 1:num_variables
        num[!, Symbol(variables[i])] = returns_matrix[:, i]
    end

    return num
end

function exponential_weighted_covariance(returns, lambda_=0.97)
    returns = convert(Matrix{Float64}, returns)
    mean_return = mean(returns, dims=1)
    normalized_returns = returns .- mean_return

    num_timesteps = size(normalized_returns, 1)
    cov_matrix = cov(returns)

    for t in 2:num_timesteps
        cov_matrix = lambda_ * cov_matrix + (1 - lambda_) * normalized_returns[t, :] * transpose(normalized_returns[t, :])
    end

    return cov_matrix
end

function calculate_EW(portfolio, prices, alpha=0.05, lambda_=0.94, portfolio_name="All")
    current_price, assets_prices, delta = prices(portfolio, prices, portfolio_name, true)
    returns = calculate(assets_prices, date_column="Date")[!, Not(:Date)]
    assets_cov = exponential_weighted_covariance(returns, lambda_)
    p_sigma = sqrt(transpose(delta) * assets_cov * delta)
    var_delta = -current_price * quantile(Normal(), alpha) * p_sigma

    return current_price[1], var_delta[1]
end

function calculate_historic(portfolio, prices, alpha=0.05, n_simulation=1000, portfolio_name="All")
    current_price, assets_prices, holdings = prices(portfolio, prices, portfolio_name)
    returns = calculate(assets_prices, date_column="Date")[!, Not(:Date)]
    assets_prices = assets_prices[!, Not(:Date)]
    sim_returns = rand(returns, n_simulation)
    sim_prices = sim_returns * transpose(assets_prices[end, :]) * holdings
    var_hist = -quantile(sim_prices, alpha) * current_price[1]

    return current_price[1], var_hist[1]
end

# Print the results
portfolio_names = ["A", "B", "C", "All"]
for portfolio_name in portfolio_names
    current_price, delta_var = calculate_EW(portfolio_data, daily_prices, portfolio_name=portfolio_name)
    current_price, hist_var, hist_sim_prices = calculate_historic(portfolio_data, daily_prices, portfolio_name=portfolio_name)
    
    println("The current value for $portfolio_name is: $current_price")
    println("VaR for $portfolio_name with Weighted Covariance is: $delta_var")
    println("VaR for $portfolio_name with Historic Simulation is: $hist_var\n")
end
