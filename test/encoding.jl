function redlograt_to_prob(x::AbstractVector)
    y = similar(x, length(x)+1)
    LogitNash.redlograt_to_prob!(y, x)
end

function prob_to_redlograt(y::AbstractVector)
    x = similar(y, length(y)-1)
    for i in eachindex(x)
        x[i] = log(y[i] / y[end])
    end
    x
end

function encoding_split_concatenated(N)
    lengths = ntuple(i -> rand(1:10), N)
    expected = ntuple(i -> rand(lengths[i]), N)
    together = vcat(expected...)

    actual = LogitNash.splitviews(together, lengths)

    @test all(actual[i] == expected[i] for i in 1:N)
end

function encoding_prob_to_log_to_prob(N)
    expected = normalize(rand(N), 1)
    redlograt = prob_to_redlograt(expected)
    actual = redlograt_to_prob(redlograt)

    @test length(redlograt) + 1 == length(expected)
    @test expected ≈ actual atol=1e-6
end

function encoding_two_probs_sanity()
    p = clamp(rand(), 1e-6, 1 - 1e-6)
    reduced = [log(p) - log(1-p)]
    expected = [p, 1-p]
    actual = redlograt_to_prob(reduced)

    @test expected ≈ actual atol=1e-6
end

@testset "Strategy encoding tests" begin
    for N in 1:5
        encoding_split_concatenated(N)
    end
    for N in 1:5
        encoding_prob_to_log_to_prob(N)
    end
    for _ in 1:10
        encoding_two_probs_sanity()
    end
end
