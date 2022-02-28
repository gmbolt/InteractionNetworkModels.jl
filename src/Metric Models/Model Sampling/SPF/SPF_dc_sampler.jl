export DcSpfMcmcSampler

struct DcSpfMcmcSampler 
    ν::Int 
    desired_samples::Int
    burn_in::Int 
    lag::Int
    par_info::Dict
    curr::Vector{Int} # Storage for current val
    prop::Vector{Int} # Storage for proposed val
    ind_del::Vector{Int} # Storage for indexing of deletions
    ind_add::Vector{Int} # Storage for indexing of additions
    vals::Vector{Int} # Storgae for new values to insert
    function DcSpfMcmcSampler(
        ;ν=4, desired_samples=1000, burn_in=0, lag=1
    )
    # req_samples = burn_in + 1 + (desired_samples - 1) * lag
    curr = Int[]
    prop = Int[]
    ind_del = zeros(Int, ν)
    ind_add = zeros(Int, ν)
    vals = zeros(Int, ν)
    par_info = Dict()
    par_info[:ν] = "(maximum number of edit operations)"
    new(
        ν, 
        desired_samples, burn_in, lag, par_info, 
        curr, prop, ind_del, ind_add, vals
        )
    end 
end 

function rand_flip(x::Int, V::UnitRange)
    tmp = rand(1:(length(V)-1))
    if tmp >= x
        return tmp+1 
    else 
        return tmp 
    end 
end 

function rand_flip!(x::Path{Int}, i::Int, V::UnitRange)
    tmp = rand(1:(length(V)-1))
    if tmp >= x[i]
        x[i] = tmp+1 
    else 
        x[i] = tmp
    end 
end 

function flip_accept_reject!(
    I_curr::Path{Int}, 
    I_prop::Path{Int}, 
    model::DcSPF, 
    mcmc::DcSpfMcmcSampler
    )

    V = model.V
    n = length(I_prop)
    n_flip = rand(1:min(mcmc.ν, n))
    k = n_flip
    i = 0 
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        # i is now index to flip
        
        I_prop[i] = rand_flip!(I_prop, i, V)
        n -= 1
        k -= 1
    end

    d, I_mode, γ = (model.d, model.mode, model.γ)
    log_α = γ * (d(I_curr, I_mode) - d(I_prop, I_mode))
    if exp(rand()) < log_α
        # Accept
        copy!(I_curr, I_prop)
        return 1
    else 
        # Reject
        copy!(I_prop, I_curr)
        return 0 
    end 
end

function rand_resize!(
    x::Path, m::Int, V::UnitRange
    )
    n = length(x)
    if n == m
        return x 
    elseif m < n
        rand_delete!(x, n-m)
    else 
        rand_insert!(x, m-n, V)
    end 
end 

function draw_sample!(
    sample_out::Vector{Path{Int}},
    mcmc::DcSpfMcmcSampler,
    model::DcSPF;
    lag::Int=mcmc.lag,
    init::Path=model.mode
    ) 

    p = model.p # length distributuion
    V = model.V
    I_curr = copy(init)
    I_prop = copy(init)

    acc_count = 0
    count = 0
    for i in eachindex(sample_out)
        n = rand(p) # Sample length 

        # Resize the paths 
        rand_resize!(I_curr, n, V)
        copy!(I_prop, I_curr)

        # Now do lag number of mcmc samples and store last
        for j in 1:lag 
            acc_count += flip_accept_reject!(
                I_curr, I_prop, model, mcmc
            )
            count += 1
        end 
        copy!(sample_out[i], I_curr) 

    end 
    return acc_count/count
end 

function draw_sample(
    mcmc::DcSpfMcmcSampler,
    model::DcSPF;
    lag::Int=mcmc.lag,
    desired_samples::Int=mcmc.desired_samples,
    init::Path=model.mode
    ) 
    sample_out = [Int[] for i in 1:desired_samples]
    a = draw_sample!(sample_out, mcmc, model, lag=lag)
    println("Acceptance prob: $a")
    return sample_out
end 