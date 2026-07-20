
function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function point_to_strat!(y, x)
    T = eltype(x)

    c = max(zero(T), maximum(x))

    ex = exp.(x .- c)
    ref = exp(-c)

    denom = ref + sum(ex)

    @inbounds for i in eachindex(x)
        y[i] = ex[i] / denom
    end

    y[end] = ref / denom

    return y
end

function point_to_strat(x)
    y = similar(x, length(x)+1)
    point_to_strat!(y, x)
end

function strat_to_point(y)
    x = Vector{eltype(y)}(undef, length(y)-1)
    for i in eachindex(x)
        x[i] = log(y[i] / y[end])
    end
    x
end
