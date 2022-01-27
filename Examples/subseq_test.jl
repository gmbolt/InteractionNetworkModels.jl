

function my_seqsample(n::Int, k::Int)
    k <= n || error("length(x) should not exceed length(a)")

    i = 0
    j = 0
    out = Int[]
    while k > 0
        u = rand()
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        push!(out, i+=1)
        n -= 1
        k -= 1
    end

    return out
end

@time my_seqsample(19,1)

x = collect(1:10)

@time rand_delete!(x, 2)

@time rand_insert!(x, 2, 1:10)

x