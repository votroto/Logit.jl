function deviations(
    payoffs::NTuple{N,<:AbstractArray{F,N}},
    x::NTuple{N,<:AbstractVector{F}}
) where {F, N}
    result = ntuple(i -> zeros(F, size(payoffs[i], i)), N)

    for i1 in axes(payoffs[i], i)
        x1 = xs[1][i1]
        for i2 in axes(payoffs[i], i)
            x2 = xs[2][i2]
            for i3 in axes(payoffs[i], i)
                x3 = xs[3][i3]
                result[1][i1] += x2*x3 * payoffs[1][i1, i2, i3]
                result[2][i2] += x1*x3 * payoffs[2][i1, i2, i3]
                result[3][i3] += x1*x2 * payoffs[3][i1, i2, i3]
            end
        end
    end
end