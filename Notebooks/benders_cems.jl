function load_inputs(inputs_path::AbstractString)

    generators = DataFrame(CSV.File(joinpath(inputs_path, "Generators_data.csv")))

    generators = select(generators, :R_ID, :Resource, :zone, :THERM, :DISP, :NDISP, :STOR, :HYDRO, :RPS, :CES,
                  :Commit, :Existing_Cap_MW, :Existing_Cap_MWh, :Cap_size, :New_Build, :Max_Cap_MW,
                  :Inv_cost_per_MWyr, :Fixed_OM_cost_per_MWyr, :Inv_cost_per_MWhyr, :Fixed_OM_cost_per_MWhyr,
                  :Var_OM_cost_per_MWh, :Start_cost_per_MW, :Start_fuel_MMBTU_per_MW, :Heat_rate_MMBTU_per_MWh, :Fuel,
                  :Min_power, :Ramp_Up_percentage, :Ramp_Dn_percentage, :Up_time, :Down_time,
                  :Eff_up, :Eff_down);

    G = generators.R_ID;

    demand_inputs = DataFrame(CSV.File(joinpath(inputs_path, "Load_data.csv")))
    VOLL = demand_inputs.Voll[1]
    S = convert(Array{Int64}, collect(skipmissing(demand_inputs.Demand_segment))) 
    nse = DataFrame(Segment=S, 
                  NSE_Cost = VOLL.*collect(skipmissing(demand_inputs.Cost_of_demand_curtailment_perMW)),
                  NSE_Max = collect(skipmissing(demand_inputs.Max_demand_curtailment)))
  
    hours_per_period = convert(Int64, demand_inputs.Hours_per_period[1])
    P = convert(Array{Int64}, 1:demand_inputs.Subperiods[1])
    W = convert(Array{Int64}, collect(skipmissing(demand_inputs.Sub_Weights)))
    T = convert(Array{Int64}, demand_inputs.Time_index)
    sample_weight = zeros(Float64, size(T,1))
    t=1
    for p in P
        for h in 1:hours_per_period
          sample_weight[t] = W[p]/hours_per_period
          t=t+1
        end
    end
  
    Z = convert(Array{Int64}, 1:3)
    demand = select(demand_inputs, :Load_MW_z1, :Load_MW_z2, :Load_MW_z3);


    variability = DataFrame(CSV.File(joinpath(inputs_path, "Generators_variability.csv")))
    variability = variability[:,2:ncol(variability)];
    
    fuels = DataFrame(CSV.File(joinpath(inputs_path, "Fuels_data.csv")));

    network = DataFrame(CSV.File(joinpath(inputs_path, "Network.csv")));

    zones = collect(skipmissing(network.Network_zones))
    lines = select(network[1:2,:], 
        :Network_lines, :z1, :z2, :z3, 
        :Line_Max_Flow_MW, :Line_Min_Flow_MW, :Line_Loss_Percentage, 
        :Line_Max_Reinforcement_MW, :Line_Reinforcement_Cost_per_MW_yr)
    lines.Line_Fixed_Cost_per_MW_yr = lines.Line_Reinforcement_Cost_per_MW_yr./20
    L = convert(Array{Int64}, lines.Network_lines);


    generators.Var_Cost = zeros(Float64, size(G,1))
    generators.CO2_Rate = zeros(Float64, size(G,1))
    generators.Start_Cost = zeros(Float64, size(G,1))
    generators.CO2_Per_Start = zeros(Float64, size(G,1))
    for g in G
        generators.Var_Cost[g] = generators.Var_OM_cost_per_MWh[g] +
            fuels[fuels.Fuel.==generators.Fuel[g],:Cost_per_MMBtu][1]*generators.Heat_rate_MMBTU_per_MWh[g]
        generators.CO2_Rate[g] = fuels[fuels.Fuel.==generators.Fuel[g],:CO2_content_tons_per_MMBtu][1]*generators.Heat_rate_MMBTU_per_MWh[g]
        generators.Start_Cost[g] = generators.Start_cost_per_MW[g] +
            fuels[fuels.Fuel.==generators.Fuel[g],:Cost_per_MMBtu][1]*generators.Start_fuel_MMBTU_per_MW[g]
        generators.CO2_Per_Start[g] = fuels[fuels.Fuel.==generators.Fuel[g],:CO2_content_tons_per_MMBtu][1]*generators.Start_fuel_MMBTU_per_MW[g]
    end

    G = intersect(generators.R_ID[.!(generators.HYDRO.==1)],G)
    G = intersect(generators.R_ID[.!(generators.NDISP.==1)],G);

    inputs = Dict{Symbol,Any}();

    inputs[:sample_weight] = sample_weight;
    inputs[:hours_per_period] = hours_per_period ;

    inputs[:G] = G;
    inputs[:S] = S;
    inputs[:P] = P;
    inputs[:T] = T;
    inputs[:Z] = Z;
    inputs[:L] = L;

    inputs[:nse] = nse         
    inputs[:generators] = generators;  
    inputs[:demand] = demand;
    inputs[:zones] = zones;       
    inputs[:lines] = lines;   
    inputs[:variability] = variability;   

    inputs[:STOR] = intersect(generators.R_ID[generators.STOR.>=1], G)
    inputs[:VRE] = intersect(generators.R_ID[generators.DISP.==1], G)
    inputs[:NEW] = intersect(generators.R_ID[generators.New_Build.==1], G)
    inputs[:OLD] = intersect(generators.R_ID[.!(generators.New_Build.==1)], G)

    return inputs

end

function scaling!(inputs::Dict{Symbol,Any})
    scaling_factor = 1e3;

    inputs[:generators].Max_Cap_MW = inputs[:generators].Max_Cap_MW/scaling_factor;
    inputs[:generators].Existing_Cap_MWh = inputs[:generators].Existing_Cap_MWh/scaling_factor;
    inputs[:generators].Fixed_OM_cost_per_MWyr = inputs[:generators].Fixed_OM_cost_per_MWyr/scaling_factor;
    inputs[:generators].Inv_cost_per_MWyr = inputs[:generators].Inv_cost_per_MWyr/scaling_factor;
    inputs[:generators].Fixed_OM_cost_per_MWhyr = inputs[:generators].Fixed_OM_cost_per_MWhyr/scaling_factor;
    inputs[:generators].Inv_cost_per_MWhyr = inputs[:generators].Inv_cost_per_MWhyr/scaling_factor;
    inputs[:generators].Var_Cost = inputs[:generators].Var_Cost/scaling_factor;

    inputs[:lines].Line_Max_Reinforcement_MW  = inputs[:lines].Line_Max_Reinforcement_MW/scaling_factor
    inputs[:lines].Line_Max_Flow_MW = inputs[:lines].Line_Max_Flow_MW/scaling_factor/scaling_factor
    inputs[:lines].Line_Fixed_Cost_per_MW_yr = inputs[:lines].Line_Fixed_Cost_per_MW_yr/scaling_factor
    inputs[:lines].Line_Reinforcement_Cost_per_MW_yr = inputs[:lines].Line_Reinforcement_Cost_per_MW_yr/scaling_factor
    
    inputs[:nse].NSE_Cost = inputs[:nse].NSE_Cost/scaling_factor

    inputs[:demand] = inputs[:demand]./scaling_factor;

    return nothing

end

function generate_monolithic_model(inputs::Dict{Symbol,Any})

    sample_weight = inputs[:sample_weight];

    hours_per_period = inputs[:hours_per_period];


    G = inputs[:G];
    S = inputs[:S];
    T = inputs[:T];
    Z = inputs[:Z];
    L = inputs[:L];

    nse = inputs[:nse];        
    generators = inputs[:generators];  
    demand = inputs[:demand]
    lines = inputs[:lines];      

    variability = inputs[:variability];

    STOR = inputs[:STOR];
    NEW = inputs[:NEW];
    OLD = inputs[:OLD];

    Expansion_Model = Model(optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm"))

    @variables(Expansion_Model, begin
        vCAP[g in G]            >= 0     # power capacity (MW)
        vRET_CAP[g in OLD]      >= 0     # retirement of power capacity (MW)
        vNEW_CAP[g in NEW]      >= 0     # new build power capacity (MW)
        
        vE_CAP[g in STOR]       >= 0     # storage energy capacity (MWh)
        vRET_E_CAP[g in intersect(STOR, OLD)]   >= 0     # retirement of storage energy capacity (MWh)
        vNEW_E_CAP[g in intersect(STOR, NEW)]   >= 0     # new build storage energy capacity (MWh)
        
        vT_CAP[l in L]          >= 0     # transmission capacity (MW)
        vRET_T_CAP[l in L]      >= 0     # retirement of transmission capacity (MW)
        vNEW_T_CAP[l in L]      >= 0     # new build transmission capacity (MW)
    end)


    for g in NEW[generators[NEW,:Max_Cap_MW].>0]
        set_upper_bound(vNEW_CAP[g], generators.Max_Cap_MW[g])
    end

    for l in L
        set_upper_bound(vNEW_T_CAP[l], lines.Line_Max_Reinforcement_MW[l])
    end

    @variables(Expansion_Model, begin
        vGEN[T,G]       >= 0  
        vCHARGE[T,STOR] >= 0 
        vSOC[T,STOR]    >= 0  
        vNSE[T,S,Z]     >= 0 
        vFLOW[T,L]     
    end);

    @constraint(Expansion_Model, cDemandBalance[t in T, z in Z], 
        sum(vGEN[t,g] for g in intersect(generators[generators.zone.==z,:R_ID],G)) +
        sum(vNSE[t,s,z] for s in S) - 
        sum(vCHARGE[t,g] for g in intersect(generators[generators.zone.==z,:R_ID],STOR)) -
        demand[t,z] - 
        sum(lines[l,Symbol(string("z",z))] * vFLOW[t,l] for l in L) == 0
    );

    @constraints(Expansion_Model, begin
        cMaxPower[t in T, g in G], vGEN[t,g] <= variability[t,g]*vCAP[g]
        cMaxCharge[t in T, g in STOR], vCHARGE[t,g] <= vCAP[g]
        cMaxSOC[t in T, g in STOR], vSOC[t,g] <= vE_CAP[g]
        cMaxNSE[t in T, s in S, z in Z], vNSE[t,s,z] <= nse.NSE_Max[s]*demand[t,z]
        cMaxFlow[t in T, l in L], vFLOW[t,l] <= vT_CAP[l]
        cMinFlow[t in T, l in L], vFLOW[t,l] >= -vT_CAP[l]
    end);

    @constraints(Expansion_Model, begin
        cCapOld[g in OLD], vCAP[g] == generators.Existing_Cap_MW[g] - vRET_CAP[g]
        cCapNew[g in NEW], vCAP[g] == vNEW_CAP[g]
        cCapEnergyOld[g in intersect(STOR, OLD)], 
            vE_CAP[g] == generators.Existing_Cap_MWh[g] - vRET_E_CAP[g]
        cCapEnergyNew[g in intersect(STOR, NEW)], 
            vE_CAP[g] == vNEW_E_CAP[g]
        cTransCap[l in L], vT_CAP[l] == lines.Line_Max_Flow_MW[l] - vRET_T_CAP[l] + vNEW_T_CAP[l]
    end);

    STARTS = 1:hours_per_period:maximum(T)        

    INTERIORS = setdiff(T,STARTS)

    @constraints(Expansion_Model, begin
        cRampUp[t in INTERIORS, g in G], 
            vGEN[t,g] - vGEN[t-1,g] <= generators.Ramp_Up_percentage[g]*vCAP[g]
        cRampUpWrap[t in STARTS, g in G], 
            vGEN[t,g] - vGEN[t+hours_per_period-1,g] <= generators.Ramp_Up_percentage[g]*vCAP[g]    
        cRampDown[t in INTERIORS, g in G], 
            vGEN[t-1,g] - vGEN[t,g] <= generators.Ramp_Dn_percentage[g]*vCAP[g] 
        cRampDownWrap[t in STARTS, g in G], 
            vGEN[t+hours_per_period-1,g] - vGEN[t,g] <= generators.Ramp_Dn_percentage[g]*vCAP[g]     
        cSOC[t in INTERIORS, g in STOR], 
            vSOC[t,g] == vSOC[t-1,g] + generators.Eff_up[g]*vCHARGE[t,g] - vGEN[t,g]/generators.Eff_down[g]
        cSOCWrap[t in STARTS, g in STOR], 
            vSOC[t,g] == vSOC[t+hours_per_period-1,g] + generators.Eff_up[g]*vCHARGE[t,g] - vGEN[t,g]/generators.Eff_down[g]
    end);

    @expression(Expansion_Model, eFixedCostsGeneration,
    sum(generators.Fixed_OM_cost_per_MWyr[g]*vCAP[g] for g in G) +
    sum(generators.Inv_cost_per_MWyr[g]*vNEW_CAP[g] for g in NEW)
    )
    @expression(Expansion_Model, eFixedCostsStorage,
    sum(generators.Fixed_OM_cost_per_MWhyr[g]*vE_CAP[g] for g in STOR) + 
    sum(generators.Inv_cost_per_MWhyr[g]*vNEW_E_CAP[g] for g in intersect(STOR, NEW))
    )
    @expression(Expansion_Model, eFixedCostsTransmission,
    sum(lines.Line_Fixed_Cost_per_MW_yr[l]*vT_CAP[l] +
    lines.Line_Reinforcement_Cost_per_MW_yr[l]*vNEW_T_CAP[l] for l in L)
    )
    @expression(Expansion_Model, eVariableCosts,
    sum(sample_weight[t]*generators.Var_Cost[g]*vGEN[t,g] for t in T, g in G)
    )
    @expression(Expansion_Model, eNSECosts,
    sum(sample_weight[t]*nse.NSE_Cost[s]*vNSE[t,s,z] for t in T, s in S, z in Z)
    )

    @objective(Expansion_Model, Min,
    eFixedCostsGeneration + eFixedCostsStorage + eFixedCostsTransmission +
    eVariableCosts + eNSECosts
    );

    return Expansion_Model


end

function generate_planning_model(inputs::Dict{Symbol,Any})

    G = inputs[:G];

    L = inputs[:L];

    generators = inputs[:generators];  
    lines = inputs[:lines];      

    STOR = inputs[:STOR];
    NEW = inputs[:NEW];
    OLD = inputs[:OLD];

    Planning_Model = Model(optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm"))

    @variables(Planning_Model, begin
        vCAP[g in G]            >= 0     # power capacity (MW)
        vRET_CAP[g in OLD]      >= 0     # retirement of power capacity (MW)
        vNEW_CAP[g in NEW]      >= 0     # new build power capacity (MW)
        
        vE_CAP[g in STOR]       >= 0     # storage energy capacity (MWh)
        vRET_E_CAP[g in intersect(STOR, OLD)]   >= 0     # retirement of storage energy capacity (MWh)
        vNEW_E_CAP[g in intersect(STOR, NEW)]   >= 0     # new build storage energy capacity (MWh)
        
        vT_CAP[l in L]          >= 0     # transmission capacity (MW)
        vRET_T_CAP[l in L]      >= 0     # retirement of transmission capacity (MW)
        vNEW_T_CAP[l in L]      >= 0     # new build transmission capacity (MW)
    end)


    for g in NEW[generators[NEW,:Max_Cap_MW].>0]
        set_upper_bound(vNEW_CAP[g], generators.Max_Cap_MW[g])
    end

    for l in L
        set_upper_bound(vNEW_T_CAP[l], lines.Line_Max_Reinforcement_MW[l])
    end


    @constraints(Planning_Model, begin
        cCapOld[g in OLD], vCAP[g] == generators.Existing_Cap_MW[g] - vRET_CAP[g]
        cCapNew[g in NEW], vCAP[g] == vNEW_CAP[g]
        cCapEnergyOld[g in intersect(STOR, OLD)], 
            vE_CAP[g] == generators.Existing_Cap_MWh[g] - vRET_E_CAP[g]
        cCapEnergyNew[g in intersect(STOR, NEW)], 
            vE_CAP[g] == vNEW_E_CAP[g]
        cTransCap[l in L], vT_CAP[l] == lines.Line_Max_Flow_MW[l] - vRET_T_CAP[l] + vNEW_T_CAP[l]
    end);

    @expression(Planning_Model, eFixedCostsGeneration,
    sum(generators.Fixed_OM_cost_per_MWyr[g]*vCAP[g] for g in G) +
    sum(generators.Inv_cost_per_MWyr[g]*vNEW_CAP[g] for g in NEW)
    )
    @expression(Planning_Model, eFixedCostsStorage,
    sum(generators.Fixed_OM_cost_per_MWhyr[g]*vE_CAP[g] for g in STOR) + 
    sum(generators.Inv_cost_per_MWhyr[g]*vNEW_E_CAP[g] for g in intersect(STOR, NEW))
    )
    @expression(Planning_Model, eFixedCostsTransmission,
    sum(lines.Line_Fixed_Cost_per_MW_yr[l]*vT_CAP[l] +
    lines.Line_Reinforcement_Cost_per_MW_yr[l]*vNEW_T_CAP[l] for l in L)
    )

    @variable(Planning_Model, vTHETA[p in inputs[:P]]>=0)

    @objective(Planning_Model, Min,
    eFixedCostsGeneration + eFixedCostsStorage + eFixedCostsTransmission + sum(vTHETA)
    );

    set_silent(Planning_Model)

    return Planning_Model


end

function generate_operation_model(inputs::Dict{Symbol,Any})

    sample_weight = inputs[:sample_weight];

    hours_per_period = inputs[:hours_per_period];

    G = inputs[:G];
    S = inputs[:S];
    T = inputs[:T];
    Z = inputs[:Z];
    L = inputs[:L];

    nse = inputs[:nse];        
    generators = inputs[:generators];  
    demand = inputs[:demand]
    lines = inputs[:lines];      

    variability = inputs[:variability];

    STOR = inputs[:STOR];

    Operation_Model =  Model(optimizer_with_attributes(HiGHS.Optimizer, "solver" => "ipm"))

    @variables(Operation_Model, begin
        vCAP[g in G]            
        vE_CAP[g in STOR] 
        vT_CAP[l in L]
    end)


    @variables(Operation_Model, begin
        vGEN[T,G]       >= 0  
        vCHARGE[T,STOR] >= 0 
        vSOC[T,STOR]    >= 0  
        vNSE[T,S,Z]     >= 0 
        vFLOW[T,L]     
    end);

    @constraint(Operation_Model, cDemandBalance[t in T, z in Z], 
        sum(vGEN[t,g] for g in intersect(generators[generators.zone.==z,:R_ID],G)) +
        sum(vNSE[t,s,z] for s in S) - 
        sum(vCHARGE[t,g] for g in intersect(generators[generators.zone.==z,:R_ID],STOR)) -
        demand[t,z] - 
        sum(lines[l,Symbol(string("z",z))] * vFLOW[t,l] for l in L) == 0
    );

    @constraints(Operation_Model, begin
        cMaxPower[t in T, g in G], vGEN[t,g] <= variability[t,g]*vCAP[g]
        cMaxCharge[t in T, g in STOR], vCHARGE[t,g] <= vCAP[g]
        cMaxSOC[t in T, g in STOR], vSOC[t,g] <= vE_CAP[g]
        cMaxNSE[t in T, s in S, z in Z], vNSE[t,s,z] <= nse.NSE_Max[s]*demand[t,z]
        cMaxFlow[t in T, l in L], vFLOW[t,l] <= vT_CAP[l]
        cMinFlow[t in T, l in L], vFLOW[t,l] >= -vT_CAP[l]
    end);

    STARTS = minimum(T):hours_per_period:maximum(T)        
    
    INTERIORS = setdiff(T,STARTS)

    @constraints(Operation_Model, begin
        cRampUp[t in INTERIORS, g in G], 
            vGEN[t,g] - vGEN[t-1,g] <= generators.Ramp_Up_percentage[g]*vCAP[g]
        cRampUpWrap[t in STARTS, g in G], 
            vGEN[t,g] - vGEN[t+hours_per_period-1,g] <= generators.Ramp_Up_percentage[g]*vCAP[g]    
        cRampDown[t in INTERIORS, g in G], 
            vGEN[t-1,g] - vGEN[t,g] <= generators.Ramp_Dn_percentage[g]*vCAP[g] 
        cRampDownWrap[t in STARTS, g in G], 
            vGEN[t+hours_per_period-1,g] - vGEN[t,g] <= generators.Ramp_Dn_percentage[g]*vCAP[g]     
        cSOC[t in INTERIORS, g in STOR], 
            vSOC[t,g] == vSOC[t-1,g] + generators.Eff_up[g]*vCHARGE[t,g] - vGEN[t,g]/generators.Eff_down[g]
        cSOCWrap[t in STARTS, g in STOR], 
            vSOC[t,g] == vSOC[t+hours_per_period-1,g] + generators.Eff_up[g]*vCHARGE[t,g] - vGEN[t,g]/generators.Eff_down[g]
    end);

    @expression(Operation_Model, eVariableCosts,
    sum(sample_weight[t]*generators.Var_Cost[g]*vGEN[t,g] for t in T, g in G)
    )
    @expression(Operation_Model, eNSECosts,
    sum(sample_weight[t]*nse.NSE_Cost[s]*vNSE[t,s,z] for t in T, s in S, z in Z)
    )

    @objective(Operation_Model, Min,
    eVariableCosts + eNSECosts
    );

    set_silent(Operation_Model)

    return Operation_Model


end

function generate_decomposed_operation_models(inputs::Dict{Symbol,Any})

    P = inputs[:P];
    hours_per_period = inputs[:hours_per_period];
    subproblems = Vector{AbstractModel}();
    for p in P
        inputs_p = deepcopy(inputs);
        inputs_p[:T] = collect((p-1)*hours_per_period + 1 : p*hours_per_period);
        subp = generate_operation_model(inputs_p);
        push!(subproblems,subp)
    end

    return subproblems
end

function solve_decomposed_operation_models(operation_models::Vector{AbstractModel},vCAP_values,vE_CAP_values,vT_CAP_values)

    operation_optval = Dict();
    λ_vCAP = Dict();
    λ_vE_CAP = Dict();
    λ_vT_CAP = Dict();

    N = length(operation_models);

    Threads.@threads for p in 1:N

        m = operation_models[p];

        fix.(m[:vCAP],vCAP_values;force=true)
        fix.(m[:vE_CAP],vE_CAP_values;force=true)
        fix.(m[:vT_CAP],vT_CAP_values;force=true)

        optimize!(m)

        operation_optval[p] = objective_value(m);

        λ_vCAP[p] = dual.(FixRef.(m[:vCAP]));
        λ_vE_CAP[p] = dual.(FixRef.(m[:vE_CAP]));
        λ_vT_CAP[p] = dual.(FixRef.(m[:vT_CAP]));

    end

    return operation_optval,λ_vCAP,λ_vE_CAP,λ_vT_CAP
end

function benders_iterations(MaxIter::Int64,planning_model::AbstractModel,operation_models::Vector{AbstractModel})

    P = collect(eachindex(operation_models));

    upper_bound = Inf;

    lower_bound = 0 
    
    tol = 1e-3;

    vCAP_best = [];
    vE_CAP_best = [];
    vT_CAP_best = [];

    println("Iteration  Lower Bound  Upper Bound          Gap")

    for k in 1:MaxIter

        ### Solve planning model
        optimize!(planning_model)

        ### Obtain new lower bound
        lower_bound = objective_value(planning_model)

        ### Obtain new capacity values
        vCAP_k = value.(planning_model[:vCAP])
        vE_CAP_k = value.(planning_model[:vE_CAP])
        vT_CAP_k = value.(planning_model[:vT_CAP])

        fix_costs = value(planning_model[:eFixedCostsGeneration]) 
                    + value(planning_model[:eFixedCostsStorage]) 
                    + value(planning_model[:eFixedCostsTransmission])

        ### Solve operational models to evaluate new capacity decisions
        operation_optval,λ_vCAP,λ_vE_CAP,λ_vT_CAP = solve_decomposed_operation_models(operation_models,vCAP_k,vE_CAP_k,vT_CAP_k)
    
        ### Update upper bound
        upper_bound_new = fix_costs + sum(operation_optval[p] for p in P)

        if upper_bound_new < upper_bound
            upper_bound = copy(upper_bound_new);
            vCAP_best = deepcopy(vCAP_k);
            vE_CAP_best = deepcopy(vE_CAP_k);
            vT_CAP_best = deepcopy(vT_CAP_k);
        end

        gap = (upper_bound-lower_bound)/lower_bound;

        print_iteration(k, lower_bound, upper_bound, gap)

        ### Check if convergence is satisfied
        if  gap > tol

            ### Add Beners cuts to planning model
            @constraint(planning_model,
                        [p in P], 
                        planning_model[:vTHETA][p] >= operation_optval[p] 
                                                    + sum(λ_vCAP[p] .* (planning_model[:vCAP] .- vCAP_k))
                                                    + sum(λ_vE_CAP[p] .* (planning_model[:vE_CAP] .- vE_CAP_k))
                                                    + sum(λ_vT_CAP[p] .* (planning_model[:vT_CAP] .- vT_CAP_k))
            )



        else

            @info "Convergence tolerance satisfied!"

            return (vCAP = vCAP_best,vE_CAP=vE_CAP_best,vT_CAP=vT_CAP_best)
        
        end
        

    end
    
    @info "The algorithm reached the maximum number of iterations"

    return nothing
    

end


function print_iteration(k, args...)
    f(x) = Printf.@sprintf("%12.4e", x)
    println(lpad(k, 9), " ", join(f.(args), " "))
    return
end