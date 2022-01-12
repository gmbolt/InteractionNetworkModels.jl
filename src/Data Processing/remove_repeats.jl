export remove_repeats!, remove_repeats

function remove_repeats!(x::Path{T}) where {T<:Union{Int,String}}
    val = 0 # Current value (entry in path) 
    tot = 0 # Rolling total entries considered 
    i = 0   # Index in new path 
    while tot < length(x)
        tmp = x[i+1]
        if tmp==val 
            deleteat!(x,i+1)
            continue 
        else 
            val = tmp
            i += 1
        end 
        tot += 1
    end
end 

function remove_repeats(x::Path{T}) where {T<:Union{Int,String}}
    y = copy(x)
    remove_repeats!(y)
    return y
end 