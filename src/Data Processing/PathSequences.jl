using DataFrames, Dates

# This contains code which makes up the pipeline of csv -> interaction seqs

export add_page_column!, get_path_sequences, my_filter


# Some intial helper functions
# Take row (vector) and output a label which merges the entries with seperator
# '-', e.g. [1,2,1] -> '1-2-1'

function make_label(X::AbstractVector; nan="NaN")
    return join(X[findall(X.!=nan)], "-")
end

function make_labels(X::AbstractArray; nan="NaN")
    Y = Vector{String}()
    for row in eachrow(X)
        push!(Y, make_label(row))
    end
    return Y
end

# tmp = rand(1:10, 100,10)
# make_labels(tmp)
# tmp1


# Now a function for dataframes

function make_labels(X::AbstractDataFrame, cols::Vector{Int}; nan="NaN")
    X = Matrix(X[:,cols])
    return make_labels(X, nan=nan)
end

function make_labels(X::AbstractDataFrame, cols::Int; nan="NaN")
    return make_labels(X, [cols], nan=nan)
end

function add_page_column!(
    X::AbstractDataFrame, 
    cols::Vector{Int}; 
    level::Int=length(cols), nan="NaN")
    # @show level
    X.page = make_labels(X, cols[1:level], nan=nan)
end


## Splitting into sessions

function get_paths(
    X::AbstractDataFrame;
    T::TimePeriod=Minute(15),
    vertex_col_name::String="vertex",
    time_col_name::String="time_gmt")

    @assert(issorted(X, time_col_name), "Must be sorted according to time.")

    function recurse_split!(
        X::AbstractDataFrame,
        Y::AbstractVector,
        ind::BitArray{1},
        i::Int;
        vertex_col_name::String)

        if findnext(ind, i) == nothing
            push!(Y, X[i:end, vertex_col_name])
            return
        else
            j = findnext(ind, i)
            # println("Path from $(i) to $(j)")
            # println(X[i:j])
            push!(Y, X[i:j, vertex_col_name])
            recurse_split!(X, Y, ind, j+1, vertex_col_name=vertex_col_name)
        end
    end
    Y = Vector{Vector{eltype(X[:,vertex_col_name])}}() # Initialise storage
    ind = diff(X[:,time_col_name]).>T
    recurse_split!(X, Y, ind, 1, vertex_col_name=vertex_col_name)
    return Y
end


## Now do on a grouped data frame (will return a vector of vector of vectors)
"""
Given a `DataFrames.GroupedDataFrame` object and some time threshold of type `Dates.TimePeriod` this will convert the data to interaction sequences. 

Column names for vertex labels and timestamps are passed names arguments.

**Note** the time column must be of type `Dates.DateTime`.
"""
function get_path_sequences(
    X::GroupedDataFrame;
    T::TimePeriod=Minute(15),
    vertex_col_name::String="vertex",
    time_col_name::String="time_gmt")

    data = Vector{Vector{Vector{String}}}(undef, length(X))
    for (i, key) in enumerate(keys(X))
        data[i] = get_paths(X[key], T=T, vertex_col_name=vertex_col_name, time_col_name=time_col_name)
    end
    return data
end


## Filtering functions
# On raw dataframe

function max_length(
    X::AbstractDataFrame;
    time_col_name::String="time_gmt",
    T::TimePeriod=Minute(15))

    tmp = diff(X[:,time_col_name]) .> T
    push!(tmp, true)
    pushfirst!(tmp, true)
    return maximum(diff(findall(tmp)))
end

function get_num_paths(
    X::AbstractDataFrame;
    time_col_name::String="time_gmt",
    T::TimePeriod=Minute(15)
    )
    return sum(diff(X[:, time_col_name]) .> T) + 1
end



function my_filter(
    X::GroupedDataFrame;
    T::TimePeriod=Minute(15),
    time_col_name::String="time_gmt",
    max_clicks=7000,
    max_path_length=750,
    max_num_paths=450
    )
    clicks = [size(x)[1] for x in X]
    path_lengths = [max_length(group, time_col_name=time_col_name, T=T) for group in X]
    num_paths = [get_num_paths(group, time_col_name=time_col_name, T=T) for group in X]

    ind = (clicks .< max_clicks) .& (path_lengths .< max_path_length) .& (num_paths .< max_num_paths)

    X[ind]
end


# On session data

function max_length(X::Vector{Vector{T}}) where T
    return maximum(map(length, X))
end

function collapsed_length(X::Vector{Vector{T}}) where T
    z = 0
    for x in X
        z += length(x)
    end
    return z
end

function my_filter(
    X::Vector{Vector{Vector{String}}};
    max_clicks=7000,
    max_path_length=750,
    max_num_paths=450
    )
    ind = (collapsed_length.(X) .< max_clicks) .& (max_length.(X) .< max_path_length) .& (length.(X) .< max_num_paths)
    return X[ind]
end
