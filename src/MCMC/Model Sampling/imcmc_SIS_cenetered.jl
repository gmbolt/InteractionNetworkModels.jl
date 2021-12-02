using Distributions

export imcmc_noisy_split, complement_rand
export mutate_delete_insert_noisy!, mutate_delete_insert_noisy
export merge_delete_insert_noisy!, merge_delete_insert_noisy

function complement_rand(
    V::UnitRange,
    x::Int
    )
    tmp = rand(V[1:(end-1)])
    if tmp ≥ x
        return tmp+1
    else 
        return tmp
    end  
end 

function mutate_delete_insert_noisy!(
    x::Path, 
    y::Path,
    ind_del::AbstractArray{Int}, 
    ind_add_x::AbstractArray{Int}, 
    ind_add_y::AbstractArray{Int}, 
    vals_x::AbstractArray{Int},
    vals_y::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )
    @show x,y
    @views for (i, index) in enumerate(ind_del)
        deleteat!(x, index - i + 1)
        deleteat!(y, index - i + 1)
    end 
    @show x,y
    for (i, val) in enumerate(x)
        if rand() > η
            if rand() < 0.5 
                println("first")
                x[i] = complement_rand(V, val)
            else 
                println("second")
                y[i] = complement_rand(V, val)
            end 
        end 

    end 

    @views for (index, val) in zip(ind_add_x, vals_x)
        # @show i, index, val
        insert!(x, index, val)
    end 
    @views for (index, val) in zip(ind_add_y, vals_y)
        # @show i, index, val
        insert!(y, index, val)
    end 

end 

function mutate_delete_insert_noisy(
    x::Path, 
    ind_del::AbstractArray{Int}, 
    ind_add_x::AbstractArray{Int}, 
    ind_add_y::AbstractArray{Int}, 
    vals_x::AbstractArray{Int},
    vals_y::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )
    y = copy(x)
    z = copy(x)
    mutate_delete_insert_noisy!(
        y, z, 
        ind_del, 
        ind_add_x, ind_add_y, 
        vals_x, vals_y, η, V
        )
    return y,z
end 

function merge_delete_insert_noisy!(
    x::Path, 
    y::Path, 
    ind_keep_x::AbstractArray{Int},
    ind_keep_y::AbstractArray{Int},
    ind_add::AbstractArray{Int},
    vals::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )

    @assert (length(x)-length(ind_del_x)) == (length(y)-length(ind_del_y)) "Invalid delition plan."

    @views for (i, index) in enumerate(ind_del_x)
        deleteat!(x, index - i + 1)
    end 
    @views for (i, index) in enumerate(ind_del_y)
        deleteat!(y, index - i + 1)
    end 
    
    # We are going to keep x and delete all of y 

    for i in eachindex(x)
        if rand() < 0.5 
            println("second")
            x[i] = popfirst!(y)
        else
            println("first")
            popfirst!(y)
        end 
    end 

    @views for (index, val) in zip(ind_add, vals)
        # @show i, index, val
        insert!(x, index, val)
    end 

end 

function merge_delete_insert_noisy(
    x::Path, 
    y::Path, 
    ind_keep_x::AbstractArray{Int},
    ind_keep_y::AbstractArray{Int},
    ind_add::AbstractArray{Int},
    vals::AbstractArray{Int},
    η::Float64,
    V::UnitRange
    )

    z = zeros(Int, length(ind_keep_x))

    for i in eachindex(z)
        if rand() < 0.5 
            println("first")
            z[i] = x[ind_keep_x[i]]
        else 
            println("second")
            z[i] = y[ind_keep_y[i]]
        end 
    end 
    @show z
    @views for (index, val) in zip(ind_add, vals)
        # @show i, index, val
        insert!(z, index, val)
    end 

    return z

end 

function imcmc_noisy_split(
    x::Path{Int}, 
    p_del::TrGeometric,
    p_ins::Geometric,
    η::Float64,
    V::UnitRange
    )
    n = length(x)
    d = rand(p_del)
    a = rand(p_ins)
    a_1 = rand(0:a)
    a_2 = a-a_1

    y = copy(x)
    z = copy(x)

    @show d, a, a_1, a_2
    ind_del = zeros(Int, d)
    StatsBase.seqsample_a!(1:n, ind_del)

    # y
    ind_add = zeros(Int, a_1)
    StatsBase.seqsample_a!(1:(n-d+a_1), ind_add)
    vals = rand(V, a_1)
    delete_insert_noisy!(y, ind_del, ind_add, vals, η, V)

    # z
    ind_add = zeros(Int, a_2)
    StatsBase.seqsample_a!(1:(n-d+a_2), ind_add)
    vals = rand(V, a_2)
    delete_insert_noisy!(z, ind_del, ind_add, vals, η, V)

    return y, z

end

function imcmc_noisy_merge(
    x::Path{Int}, 
    y::Path{Int},
    δ::Int,
    V::AbstractVector
    )
    n₁ = length(x)
    n₂ = length(y)
    if n₁ < n₂ 
        d₁ = rand(0:min(δ,n₁))
        d₂ = n₂ - n₁ + d₁
    else    
        d₂ = rand(0:min(δ,n₂))
        d₁ = n₁ - n₂ + d₂
    end 
    
    d = rand(0:min(δ,n))
    a₁ = rand(0:(δ-d))
    a₂ = δ - d - a₁

    @show d, a₁, a₂
    y = copy(x)
    z = copy(x)

    ind_del = zeros(Int, d)
    StatsBase.seqsample_a!(1:n, ind_del)

    # y
    ind_add = zeros(Int, a₁)
    StatsBase.seqsample_a!(1:(n-d+a₁), ind_add)
    vals = rand(V, a₁)
    delete_insert!(y, ind_del, ind_add, vals)

    # z
    ind_add = zeros(Int, a₂)
    StatsBase.seqsample_a!(1:(n-d+a₂), ind_add)
    vals = rand(V, a₂)
    delete_insert!(z, ind_del, ind_add, vals)

    return z

end