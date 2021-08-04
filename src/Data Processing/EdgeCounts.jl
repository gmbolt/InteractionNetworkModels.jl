using DataFrames, Dates

function get_edge_counts(
    X::GroupedDataFrame;
    vertex_col_name::String="page",
    time_col_name::String="time_gmt"
    )
    @assert eltype(df[:,time_col_name]) <: DateTime "Time information must be in DateTime format"

    for group in df_g
        for i = 1:(size(group)[1]-1)
            if i
            end 
        end 
    end 
end

eltype(df[:,"time_gmt"]) <: DateTime

g = path_graph(10)

typeof(g)

G1 = SimpleDiGraph([0 10; 1 0])

gplot(g)

for e in edges(G1)
    print(e)
end

add_edge!(g, 3, 2)

size(df_g[1])

G2 = MetaGraph(g, 1)
add_edge!(G2, 1,2)
gplot(G2)

for e in edges(G2)
    println(e)
end

m = Multiset(1,2,1,1,2)

using Multigraphs

g = Multigraph(10)

add_edge!(g, 1,10)

println(g)

for e in edges(g)
    println(e)
end

m = Multiset{Pair}()

push!(m, Pair("a", "b"))

m.data

Pair("a", "b")
