export delete_insert!



function accept_reject!(
    x_curr::Vector{Int},
    x_prop::Vector{Int},
    mcmc::SnfMcmcInsertDelete,
    model::SNF{Int}
    )
    M_curr = sum(x_curr)
    M_prop = M_curr
    δ = rand(1:mcmc.ν)
    d = rand(0:min(δ,M)) # Number of edges to delete
    log_ratio = 0.0 
    
    # Delete edges 
    for i in 1:d 
        counter = 0
        z = rand(1:M_prop)
        for (edge, edge_weight) in enumerate(x_prop)
            counter += edge_weight
            if counter ≥ z 
                x_prop[edge] -= 1
                M_prop -= 1
                break
            end 
        end 
    end 
    # Add edges 
    N = length(x_curr)
    for i in 1:(δ-d)
        x_prop[rand(1:N)] += 1
    end 


    



end 