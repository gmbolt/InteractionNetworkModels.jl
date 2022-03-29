export string_to_int, int_to_string

# This file contains some helper functions to map between interaction sequences with String and Int values.
# This is helpful for inference, where we will typically map to integers to run posterior samplers and then map back after.

"""
    string_to_int(::InteractionSequence{String}, ::Dict{String,Int})

    Takes `S::InteractionSequence{String}` and outputs a value of type `InteractionSequence{Int}` by applying the map encoded via the dictionary `mapper::Dict{String,Int}`.
"""
function string_to_int(
    S::InteractionSequence{String},
    mapper::Dict{String,Int}
    )
    out = similar.(S, Int)
    for i in eachindex(S)
        tmp_in = S[i]
        tmp_out = out[i]
        for j in eachindex(tmp_out)
            tmp_out[j] = mapper[tmp_in[j]]
        end 
    end 
    return out
end 

function string_to_int(
    data::InteractionSequenceSample{String},
    mapper::Dict{String,Int}
    )
    return map(x -> string_to_int(x, mapper), data)
end 

function string_to_int(
    data::InteractionSequence{String}
    )

    mapper = Dict{String,Int}()
    i = 1
    for v in vcat(data...)
        if v ∈ keys(mapper)
            continue
        else 
            mapper[v] = i 
            i +=1 
        end 
    end 
    mapper_inv = Dict(v => k for (k,v) in mapper)
    return string_to_int(data, mapper), mapper_inv
end 

function string_to_int(
    data::InteractionSequenceSample{String}
    )

    mapper = Dict{String,Int}()
    i = 1
    for v in vcat([vcat(S...) for S in data]...)
        if v ∈ keys(mapper)
            continue
        else 
            mapper[v] = i 
            i +=1 
        end 
    end 
    iter = zip(keys(mapper), values(mapper))
    mapper_inv = Dict(v => k for (k,v) in iter)
    return string_to_int(data, mapper), mapper_inv
end 

function int_to_string(
    S::InteractionSequence{Int}, 
    mapper::Dict{Int,String}
)

    out = similar.(S, String)
    for i in eachindex(S)
        tmp_in = S[i]
        tmp_out = out[i]
        for j in eachindex(tmp_out)
            tmp_out[j] = mapper[tmp_in[j]]
        end 
    end 
    return out
end 

function int_to_string(
    data::InteractionSequenceSample{Int},
    mapper::Dict{Int,String}
    )
    return map(x -> int_to_string(x, mapper), data)
end 

function int_to_string(
    data::InteractionSequenceSample{Int}
    )
    mapper = Dict{Int,String}()
    for v in vcat([vcat(S...) for S in data]...)
        if v ∈ keys(mapper)
            continue
        else 
            mapper[v] = string(v)
        end 
    end 
    iter = zip(keys(mapper), values(mapper))
    mapper_inv = Dict(v => k for (k,v) in iter)
    return int_to_string(data, mapper), mapper_inv
end 

function string_to_int(x::Vector{String})
    out = Int[]
    count = 1
    mapper = Dict{String,Int}()
    for v in x
        if v ∉ keys(mapper)
            mapper[v] = count
            count += 1
        end 
        push!(out, mapper[v])
    end
    mapper_inv = Dict(v=>k for (k,v) in mapper)
    return out, mapper_inv    
end 