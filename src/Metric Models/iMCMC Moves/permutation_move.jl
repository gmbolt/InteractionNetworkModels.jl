using Random

export PathPermutationMove

struct PathPermutationMove <: InvMcmcMove 
    ν::Int 
    counts::Vector{Int}  # Track acceptance 
    function PathPermutationMove(;ν::Int=3)
        new(
            ν, 
            [0,0]
        )
    end 
end 

Base.show(io::IO, x::PathPermutationMove) = print(io, "PathPermutationMove(ν=$(x.ν))")

# Exclusive shuffle (without identity permutation )
function excl_shuffle!(x)
    if length(x) > 1
        ind_ref = collect(1:length(x))
        ind = randperm(length(x))
        while ind == ind_ref
            shuffle!(ind)
        end 
        permute!(x, ind)
    end 
end 

function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::PathPermutationMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
    )

    n = length(S_curr)
    n_var = n 
    ν = move.ν
    k = rand(1:min(n,ν)) # Number of paths to permute
    i = 0
    # Now we use alg which samples random subseq (for index of paths to permute)
    while k > 0
        u = rand()
        q = (n_var - k) / n_var
        while q > u  # skip
            i += 1
            n_var -= 1
            q *= (n_var - k) / n_var
        end
        i+=1
        # i is now index of path to permute
        # @inbounds ind[j] = i
        @inbounds excl_shuffle!(S_prop[i])
        n_var -= 1
        k -= 1
    end
    log_ratio = 0.0 # In this case the proposal is symmetric
    return log_ratio

end 