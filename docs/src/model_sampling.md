# Model Sampling 

## Defining Models 

## Defining Samplers 

## Obtaining Samples

### 1: Calling Sampler Types

Any object of type `SisMcmcSampler` or `SimMcmcSampler` can be called like a function, taking as input an SIS (or SIM) model and outputting an MCMC sample. In this case, the sample is returned as part of a specialsed MCMC output type. This is useful traceplots are desired, since one can simply call `plot()` on the mcmc output. 



### 2: Using `draw_sample()`

```@docs 
draw_sample
```

### 3: In-Place Sampling

```@docs
draw_sample!
```

## Plotting 
