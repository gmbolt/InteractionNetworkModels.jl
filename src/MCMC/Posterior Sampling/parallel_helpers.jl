export get_split, get_split_inds, get_split_ind_iters

function get_split(
    n::Int, bins::Int   
    )
    increment = n / bins 
    rem = 0.0
    tot = 0
    step = floor(Int, increment + rem)
    out = Int[]
    for i in 1:(bins-1)
        step = floor(Int, increment + rem)
        rem += increment - step
        push!(out, step) 
        tot += step
    end 
    push!(out, n-tot)
end


function get_split_inds(
    n::Int, bins::Int   
    )
    split = get_split(n, bins)
    pushfirst!(split, 1)
    cumsum(split)
end 

function get_split_ind_iters(n::Int, bins::Int)
    split_inds = get_split_inds(n, bins)
    [split_inds[i]:(split_inds[i+1]-1) for i in 1:(length(split_inds)-1)]
end 