

function fd_jacobian(f, _x; h = 1e-6)
    x = copy(_x)
    fx = f(x)
    m = length(fx)
    n = length(x)
    J = zeros(eltype(fx), m, n)

    for j in 1:n
        x[j] += h
        J[:, j] = (f(x) .- fx) ./ h
        x[j] -= h
    end

    return J
end

function fast_lu_sanity()
    original = randn(5, 5)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 5)
    actual = copy(original)

    lu_result = lu(original)
    @test @allocated(LogitNash.fast_lu!(actual, ipiv)) == 0

    @test actual ≈ lu_result.factors
    @test ipiv == lu_result.ipiv
end

function fast_lu_singular()
    original = zeros(5, 5)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 5)
    @test_throws SingularException LogitNash.fast_lu!(original, ipiv)
end

function residual_wrapper(x, t, utils::NTuple{N}) where {N}
    rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))

    pi = ntuple(i -> Vector{Float64}(undef, size(utils[i], i)), Val(N))
    res = zeros(rsize)
    ubar = ntuple(i -> zeros(size(utils[i], i)), Val(N))

    mu = LogitNash.splitviews(x, size(first(utils)) .- 1)
    LogitNash.redlograt_to_prob!.(pi, mu)

    LogitNash.unilateral_deviations!(ubar, utils, pi)
    LogitNash.residual!(res, mu, ubar, x, t, utils)
end

function jacobian_x_analytic_vs_finitedifference()
    utils = (randn(2,3,4), randn(2,3,4), randn(2,3,4))

    t = rand()*100

    _pi = ntuple(i->normalize(rand(size(utils[i], i)),1), 3)
    x = vcat(prob_to_redlograt.(_pi)...)

    wsdudpi = ntuple(p -> ntuple(q -> zeros(size(utils[p], p), size(utils[p], q)), 3), 3)
    wsFx = zeros(length(x), length(x))

    LogitNash.unilateral_derivatives!(wsdudpi, utils, _pi)
    LogitNash.jacobian_x!(wsFx, _pi, t, wsdudpi, utils)

    Fx_fd = fd_jacobian(x -> residual_wrapper(x, t, utils), x)

    @test norm(wsFx - Fx_fd) <= 1e-4
end

function jacobian_t_analytic_vs_finitedifference()
    utils = (randn(2,3,4), randn(2,3,4), randn(2,3,4))

    t = rand()*100

    _pi = ntuple(i->normalize(rand(size(utils[i], i)),1), 3)
    x = vcat(prob_to_redlograt.(_pi)...)

    rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))
    wsubar = ntuple(i -> zeros(size(utils[i], i)), 3)
    wsFt = Vector{Float64}(undef, rsize)

    mu = LogitNash.splitviews(x, size(first(utils)) .- 1)

    LogitNash.unilateral_deviations!(wsubar, utils, _pi)

    LogitNash.jacobian_t!(wsFt, wsubar, mu, utils)

    Ft_fd = fd_jacobian(t -> residual_wrapper(x, only(t), utils), [t])

    @test norm(wsFt - Ft_fd) <= 1e-4

end

function residual_zero_only_at_uniform()
    utils = (randn(2,3,4), randn(2,3,4), randn(2,3,4))

    nx = sum(size(utils[i], i) - 1 for i in eachindex(utils))
    x_zero = zeros(nx)
    x_rand = randn(nx)

    res_zero = residual_wrapper(x_zero, 0, utils)
    res_rand = residual_wrapper(x_rand, 0, utils)

    @test res_zero ≈ zero(res_zero) atol=1e-4
    @test norm(res_rand, Inf) >= 1e-4
end

@testset "Main sub-routines work as expected" begin
    for N in 1:5
        fast_lu_sanity()
    end
    for N in 1:5
        jacobian_x_analytic_vs_finitedifference()
    end
    for N in 1:5
        jacobian_t_analytic_vs_finitedifference()
    end
    for N in 1:5
        residual_zero_only_at_uniform()
    end
    fast_lu_singular()
end
