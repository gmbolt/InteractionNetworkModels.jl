using InteractionNetworkModels

move = PathPermutationMove(ν=2)

V = 1:10
S_curr = [rand(V, i) for i in 1:5]
S_prop = deepcopy(S_curr)
pointers = [Int[] for i in 1:100]

prop_sample!(S_curr, S_prop, move, pointers, V)

S_curr
S_prop