function splitviews(x::AbstractVector, js::NTuple{N,Int}) where {N}
    offs = cumsum((0, js...))
    ntuple(i -> @view(x[(offs[i]+1):offs[i+1]]), N)
end

function redlograt_to_prob!(y::AbstractVector{F}, x::AbstractVector{F}) where F
    c = maximum(x, init=zero(F))

    denom = zero(F)
    @inbounds for i in eachindex(x)
        v = exp(x[i] - c)
        y[i] = v
        denom += v
    end
    y[end] = exp(-c)
    denom += y[end]

    @inbounds for i in eachindex(y)
        y[i] = y[i] / denom
    end

    return y
end
