using InteractionNetworkModels

mode = rand(0:1,10,10)
model = CER(mode, 0.1, true)

draw_sample(model, )