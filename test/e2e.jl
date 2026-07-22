function util2d(payoffs::NTuple{2}, a::NTuple{2})
    pa1 = dot(a[1], payoffs[1], a[2])
    pa2 = dot(a[1], payoffs[2], a[2])

    [pa1, pa2]
end

function coordination_risk_dominant_eq(payoffs::NTuple{2})
    (A, B) = payoffs

    risk_ul = (A[1]-A[2])*(B[1]-B[3])
    risk_dr = (A[4]-A[3])*(B[4]-B[2])

    if isapprox(risk_dr, risk_ul)
        p_row = (B[4]-B[2])/(B[1]-B[3]-B[2]+B[4])
        p_col = (A[4]-A[3])/(A[1]-A[3]-A[2]+A[4])
        return ([p_row, 1-p_row], [p_col, 1-p_col])
    elseif risk_ul >= risk_dr
        return ([1, 0], [1, 0])
    else
        return ([0, 1], [0, 1])
    end
end

function e2e_battle_of_the_sexes()
    A::Matrix{Float64} = [3 0; 0 2]
    B::Matrix{Float64} = [2 0; 0 3]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = coordination_risk_dominant_eq(payoffs)
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_coordination_wiki()
    A::Matrix{Float64} = [2 1; 1 2]
    B::Matrix{Float64} = [4 3; 3 4]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = coordination_risk_dominant_eq(payoffs)
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_coordination_pure()
    A::Matrix{Float64} = [8 0; 0 8]
    B::Matrix{Float64} = [8 0; 0 8]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = coordination_risk_dominant_eq(payoffs)
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_coordination_assurance()
    A::Matrix{Float64} = [8 0; 0 5]
    B::Matrix{Float64} = [8 0; 0 5]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = coordination_risk_dominant_eq(payoffs)
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_stag_hunt()
    A::Matrix{Float64} = [8 0; 7 5]
    B::Matrix{Float64} = [8 7; 0 5]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = coordination_risk_dominant_eq(payoffs)
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_prisonners_dilemma()
    A::Matrix{Float64} = [5 0; 8 1]
    B::Matrix{Float64} = [5 8; 0 1]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = ([0, 1], [0, 1])
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_matching_pennies()
    A::Matrix{Float64} = [1 -1; -1 1]
    B::Matrix{Float64} = [-1 1; 1 -1]
    payoffs = (A, B)
    pi, status = nash(payoffs)

    expected_profile = ([0.5, 0.5], [0.5, 0.5])
    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, expected_profile)
    @test actual ≈ expect atol=1e-4
end

function e2e_guessing()
    util1::Matrix{Float64} = [0 1 4; 1 0 1; 4 1 0]
    payoffs = (util1, -util1)
    pi, status = nash(payoffs)

    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, ([0.5, 0.0, 0.5], [0.0, 1.0, 0.0]))
    @test actual ≈ expect atol=1e-4
end

function e2e_constant()
    payoffs = (zeros(3, 3), zeros(3, 3))
    pi, status = nash(payoffs)

    actual = util2d(payoffs, pi)
    expect = util2d(payoffs, ([1/3, 1/3, 1/3], [1/3, 1/3, 1/3]))
    @test actual ≈ expect atol=1e-4
end

@testset "Basic games with known solutions" begin
    e2e_battle_of_the_sexes()
    e2e_constant()
    e2e_coordination_assurance()
    e2e_coordination_pure()
    e2e_coordination_wiki()
    e2e_guessing()
    e2e_matching_pennies()
    e2e_prisonners_dilemma()
    e2e_stag_hunt()
end