export draw_sample!, draw_sample

# We can sample from the CER model exactly 

function draw_sample!(
    out::Vector{Matrix},
    model::CER
    )
    for A in out 
        copy!(A, model.mode)
        for c in eachcol(A)
            for i in eachindex(c)
                if rand() < model.α
                    c[i] = (c[i] + 1) % 2
                end 
            end 
        end 

    end 
end 

function draw_sample_no_copy!(
    out::Vector{Matrix},
    model::CER
    )
    for A in out 
        for c in eachcol(A)
            for i in eachindex(c)
                if rand() < model.α
                    c[i] = (c[i] + 1) % 2
                end 
            end 
        end 
    end 
end 

function draw_sample(
    model::CER,
    n::Int
    )
    out = fill(copy(A), n)
    draw_sample_no_copy!(out, model)
    return out
end 

# The SNF in general requires some MCMC 

