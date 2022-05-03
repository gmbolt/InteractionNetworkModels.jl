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


function prop_sample!(
    S_curr::InteractionSequence{Int},
    S_prop::InteractionSequence{Int},
    move::PathPermutationMove,
    pointers::InteractionSequence{Int},
    V::UnitRange
    )

    n = length(S_curr)
    ν = move.ν
    k = rand(1:min(n,ν)) # Number of paths to permute
    i = 0
    # Now we use alg which samples random subseq (for index of paths to permute)
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        i+=1
        # i is now index of path to permute
        # @inbounds ind[j] = i
        @inbounds shuffle!(S_prop[i])
        n -= 1
        k -= 1
    end
    log_ratio = 0.0 # In this case the proposal is symmetric
    return log_ratio

end 