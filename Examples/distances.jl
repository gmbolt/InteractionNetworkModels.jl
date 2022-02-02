using InteractionNetworkModels, BenchmarkTools, Distances


d_lcs = LCS()

x,y = (rand(1:3,10), rand(1:3, 10))

@btime d(x,y)

d_fast = FastLCS(100)
@btime d_fast(x,y)

d = MatchingDist(LCS())
d_fast = FastMatchingDist(LCS(), 50)
x,y = (

    [rand(1:3, rand(4:6)) for i in 1:rand(4:5)], 
    [rand(1:3, rand(4:6)) for i in 1:rand(4:5)]
)
C = zeros(100,100)
@btime d(x,y)
@btime d_fast(x,y)
@btime matching_dist_with_memory!(x,y,d_lcs,C)

@btime pairwise_inbounds(d_lcs, x,y)
@btime Distances.pairwise(d_lcs, x,y)

d = MatchingDist(LCS())
d_norm = Normalised(d)

d_norm([[1,2,1]], [[1,2,3,4,5,4],[1]])

typeof(Normalised(d)) <: InteractionSetDistance

supertype(typeof(Normalised(d)))
supertype(typeof(Euclidean()))