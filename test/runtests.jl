using Test
using LogitNash

function e2e_guessing_game()
    util::Matrix{Float64} = [0 1 4; 1 0 1; 4 1 0]
    pi, status = nash((util, -util))

    @test isapprox(pi[1], [0.5, 0.0, 0.5], atol=1e-4)
    @test isapprox(pi[2], [0.0, 1.0, 0.0], atol=1e-4)
end

e2e_guessing_game()