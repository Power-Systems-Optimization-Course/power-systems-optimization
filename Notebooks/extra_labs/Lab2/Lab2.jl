using Gurobi
include("Lab2_code.jl")

# REPLACE THIS WITH YOUR PATH TO INPUT DATA:
inputs_directory = "/Users/gabrielmantegna/Documents/GitHub/power-systems-optimization/Notebooks/complex_expansion_data"
inputs_path = joinpath(inputs_directory, "10_days")

# REPLACE THIS WITH YOUR PATH TO WHERE YOU WANT TO OUTPUT DATA:
outputs_directory = "/Users/gabrielmantegna/Documents/GitHub/power-systems-optimization/Notebooks/Lab2_results"
if !(isdir(outputs_directory))
	mkdir(outputs_directory)
end

### STANDARD MODEL ###########

# load inputs
(generators, G) = load_input_generator(inputs_path)
(demand, nse, sample_weight, hours_per_period, P, S, W, T, Z) = load_input_demand(inputs_path)
variability = load_input_variability(inputs_path)
(lines, L) = load_input_network(inputs_path);

# solve model
(generator_results,
	storage_results,
	transmission_results,
	nse_results,
	cost_results,
	time_10_days) = solve_cap_expansion(generators, G,
	demand, S, T, Z,
	nse,
	sample_weight, hours_per_period,
	lines, L,
	variability, Gurobi.Optimizer)

# report time
time_10_days = round(time_10_days, digits = 2)
"Time elapsed to solve with 10 sample days: $time_10_days seconds"

# write results
write_results(generator_results,
	storage_results,
	transmission_results,
	nse_results,
	cost_results,
	time_10_days,
	joinpath(outputs_directory, "10_days_Solution"))

####################

#### MODEL WITH UNIT COMMITMENT #####

# # solve model with UC
# (generator_results,
# storage_results,
# transmission_results,
# nse_results,
# cost_results,
# time_10_days_UC) = solve_cap_expansion_uc(generators, G,
# demand, S, T, Z,
# nse,
# sample_weight, hours_per_period,
# lines, L,
# variability, Gurobi.Optimizer, 0.01)

# # report time
# time_10_days_UC = round(time_10_days_UC, digits = 2)
# "Time elapsed to solve with 10 sample days: $time_10_days_UC seconds"

# # write results
# write_results(generator_results,
# 	storage_results,
# 	transmission_results,
# 	nse_results,
# 	cost_results,
# 	time_10_days_UC,
# 	joinpath(outputs_directory, "10_days_Solution_UC"))

#########