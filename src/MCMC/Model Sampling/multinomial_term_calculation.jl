export log_multinomial_ratio, myaddcounts!

function myaddcounts!(d::Dict{T}, s::T) where {T}
    d[s] = get(d, s, 0) + 1
end 

# """
# For two vectors ``X = [x_1, \dots, x_N]`` and ``Y = [y_1, \dots, y_N]`` this calculates the following term
# ```math
#     log \left( \frac{N! w^\prime_1! \cdots w^\prime_{m}!}{M! w_1!\cdots w_{n}!}\right)
# ```
# where ``w_i`` and ``w_j^\prime`` represent the counts of (respectively) ``n`` and ``m`` unique entries of ``X`` and ``Y``. 

# This comes into use when looking to sample multisets via marginalising out order information of sequences. 
# """
function log_multinomial_ratio(x::AbstractVector, y::AbstractVector)
    if length(x) > length(y)
        return -log_multinomial_ratio(y,x)
    end 
    # Now we can assume length(x) ≤ length(y)
    z = 0.0
    dictx = Dict{eltype(x), Int}()  # To store values seen in x and their counts
    dicty = Dict{eltype(y), Int}()  # To store values seen in y and their counts

    # First sort the element counts terms
    for i=1:length(x) 
        # @show x_val, y_val, typeof(x_val)
        @inbounds myaddcounts!(dictx, x[i])
        @inbounds myaddcounts!(dicty, y[i])
        @inbounds z = z + log(dicty[y[i]]) - log(dictx[x[i]]) 
    end 
    for i=(length(x)+1):(length(y))
        @inbounds myaddcounts!(dicty, y[i])
        @inbounds z = z + log(dicty[y[i]])
    end 
    # Now the object size terms (log())
    for i=(length(x)+1):length(y)
        z = z - log(i)
    end 
    return z
end 

# Test instances

# tmp1 = [1, 1, 1, 2, 2]
# tmp2 = [1, 1, 2, 2, 2]
# @time log_multinomial_ratio(tmp1, tmp2)

# tmp1 = [1, 2, 3]
# tmp2 = [1, 2, 2, 3, 3, 3]
# @time log_multinomial_ratio(tmp1, tmp2)
# -log(10)