using JuMP
using HiGHS


# function to calculate upper and lower bound of variable x
function compute_xbounds()
    A = [-0.10 1.0; 0.35 -0.17]
    b = [8.4,-1.4]
    x = A \ b
    convert.(Int64,round(x))
end

# function to build a JuMP model with a specific optimizer
function build_model(optimizer)
    Model = Model()
    set_optimizer(Model, optimizer)
    return Model
end

# function to define the variables x and y of the JuMP model and set upper and lower bound of x
function set_variables(model::GenericModel{Float64}, x_lower_bound::Float64, x_upper_bound::Float64)
    @variable(model, x)
    @variable(model, y)

    set_lower_bound(x, x_lower_bound)
    set_upper_bound(x, x_upper_bound)

    return (x,y)
end

# function to define the constraints of the JuMP model
function set_constraints(model::GenericModel{Float64}, x::VariableRef, y::VariableRef)
    @constraints(model, 
    begin
        time_constraint, 10x + 8y <= 80
        materials_constraint, 7x + 11y <= 77    
    end)
end

# function to define the objective of the JuMP model
function set_objective(model::GenericModel{Float64}, x::VariableRef, y::VariableRef)
    @expression(model, objective, 150x + 175y)
    @objective(model, max, objective)
end


# workflow
x_lower_bound, x_upper_bound = compute_xbounds()    # x-bounds calculation
model = build_model(HiGHS.Optimizer)                # JuMP model definition
x, y = set_variables(model, x_lower_bound, x_upper_bound)   # variables definition
set_constraints(model, x, y)    # constraints definition
set_objective(model, x, y)      # objective definition
optimize!(model)                # model optimization 
print(model)