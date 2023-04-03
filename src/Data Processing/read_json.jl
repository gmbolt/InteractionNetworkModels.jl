using JSON
export read_inter_seq_json

function read_inter_seq_json(filepath::String)
    open(filepath, "r") do f
        stringdata = readall(f)  
        out_any = JSON.parse(dicttxt)  # Will be of type Vector{Anu}
    end 
    out = Vector{Vector{String}}[]
    tmp_obs = Vector{String}[]
    for obs in out_any
        empty!(tmp_obs)
        for path in obs 
            push!(tmp_obs, convert(Vector{String}, path))
        end 
        push!(tmp_obs, out)
    end 
    return out 
end 