using InteractionNetworkModels

d1 = LSP()

x = [1,1,2,1,2,1,2]
y = [1,1,2,1,3,4,5]

@time d1(x,y)

d2 = FastLSP(100)

@time d2(x,y)

