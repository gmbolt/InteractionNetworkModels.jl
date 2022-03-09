using RecipesBase

# # Plot Recipes 
@recipe function f(output::SnfMcmcOutput)
    model = output.model
    sample = output.sample
    x = map(x -> model.d(model.mode, x), sample)
    xguide --> "Sample"
    yguide --> "Distance from Mode"
    legend --> false
    size --> (800, 300)
    margin --> 5mm
    x
end 