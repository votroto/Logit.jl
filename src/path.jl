include("turocy.jl")
include("spaces.jl")
using LinearAlgebra
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: liblapack

function fast_lu!(A::Matrix{Float64}, ipiv::Vector{BlasInt})
    m, n = size(A)
    info = Ref{BlasInt}()
    # Direct Fortran call to dgetrf (Double precision GETRF)
    ccall((@blasfunc(dgetrf_), liblapack), Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64},
            Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
        m, n, A, max(1, m), ipiv, info)

    if info[] > 0
        throw(SingularException(info[]))
    elseif info[] < 0
        throw(ArgumentError("LAPACK dgetrf error"))
    end
    return A
end

_zero_nested!(x::AbstractArray) = fill!(x, zero(eltype(x)))
function _zero_nested!(t::Tuple)
    for ti in t
        _zero_nested!(ti)
    end
end

# allocate all scratch storage once
function make_hc_workspace(x_template::Vector{Float64}, utils::NTuple{N}) where {N}
    T = eltype(x_template)
    n = length(x_template)
    rsize = sum(size(first(utils), i) - 1 for i in eachindex(utils))

    pi = ntuple(i -> Vector{T}(undef, size(utils[i], i)), Val(N))
    res = Vector{T}(undef, rsize)
    ubar = ntuple(i -> zeros(T, size(utils[i], i)), Val(N))
    dudpi = ntuple(p -> ntuple(q -> zeros(T, size(utils[p], p), size(utils[p], q)), Val(N)), Val(N))

    Fx = zeros(T, n, n)
    Ft = Vector{T}(undef, rsize)
    ipiv = Vector{BlasInt}(undef, n)

    v = Vector{T}(undef, rsize)
    w = Vector{T}(undef, rsize)

    dxdt = Vector{T}(undef, rsize)
    xpred = Vector{T}(undef, n)
    x_diff = Vector{T}(undef, n)
    dx_step = Vector{T}(undef, n)
    x_nxt = Vector{Float64}(undef, n)

    return (; pi, res, ubar, dudpi, Fx, ipiv, Ft, v, w, dxdt, xpred, x_diff, dx_step, x_nxt)
end

function predict!(
    dx_out::Vector{Float64},
    x::Vector{Float64},
    t::Float64,
    lastdx::Vector{Float64},
    lastdt::Float64,
    utils::NTuple{N},
    ws
) where {N}
    _zero_nested!(ws.ubar)
    _zero_nested!(ws.dudpi)

    mu = splitviews(x, size(first(utils)) .- 1)
    redlograt_to_prob!.(ws.pi, mu)

    unilateral_derivatives!(ws.dudpi, utils, ws.pi)
    jacobian_x!(ws.Fx, ws.pi, t, ws.dudpi, utils)

    unilateral_deviations!(ws.ubar, utils, ws.pi)
    jacobian_l!(ws.Ft, ws.ubar, mu, t, utils)

    # In-place factorization and solve (0 allocations)
    fast_lu!(ws.Fx, ws.ipiv)

    copyto!(ws.dxdt, ws.Ft)
    LinearAlgebra.LAPACK.getrs!('N', ws.Fx, ws.ipiv, ws.dxdt)

    dtds = 1.0 / sqrt(1.0 + dot(ws.dxdt, ws.dxdt))

    # Evaluate direction check before mutating dx_out (in case dx_out aliases lastdx)
    sign_check = -dtds * dot(ws.dxdt, lastdx) + dtds * lastdt

    if sign_check < 0.0
        @. dx_out = dtds * ws.dxdt
        dtds = -dtds
    else
        @. dx_out = -dtds * ws.dxdt
    end

    return dx_out, dtds
end

function correct!(
    xlast::Vector{Float64},
    tlast::Float64,
    dx::Vector{Float64},
    dt::Float64,
    ds::Float64,
    utils::NTuple{N},
    ws;
    iters::Int=3,
    abs_tol::Float64=1e-6,
    rel_tol::Float64=1e-12
) where {N}
    @. ws.xpred = xlast + ds * dx
    tpred = tlast + ds * dt

    copyto!(ws.x_nxt, ws.xpred)
    t_out = tpred

    i = 0
    while true
        fill!(ws.res, 0.0)
        _zero_nested!(ws.ubar)

        mu = splitviews(ws.x_nxt, size(first(utils)) .- 1)
        redlograt_to_prob!.(ws.pi, mu)

        unilateral_deviations!(ws.ubar, utils, ws.pi)
        residual!(ws.res, mu, ws.ubar, ws.x_nxt, t_out, utils)

        @. ws.x_diff = ws.x_nxt - ws.xpred
        r_con = dot(ws.x_diff, dx) + (t_out - tpred) * dt

        # Absolute convergence check
        if dot(ws.res, ws.res) + r_con^2 < abs_tol^2
            return true, ws.x_nxt, t_out
        end

        _zero_nested!(ws.dudpi)

        unilateral_derivatives!(ws.dudpi, utils, ws.pi)
        jacobian_x!(ws.Fx, ws.pi, t_out, ws.dudpi, utils)
        jacobian_l!(ws.Ft, ws.ubar, mu, t_out, utils)

        @. ws.res = -ws.res

        fast_lu!(ws.Fx, ws.ipiv)

        copyto!(ws.v, ws.Ft)
        LinearAlgebra.LAPACK.getrs!('N', ws.Fx, ws.ipiv, ws.v)

        copyto!(ws.w, ws.res)
        LinearAlgebra.LAPACK.getrs!('N', ws.Fx, ws.ipiv, ws.w)

        dt_step = (-r_con - dot(dx, ws.w)) / (dt - dot(dx, ws.v))
        @. ws.dx_step = ws.w - dt_step * ws.v

        step_norm = sqrt(dot(ws.dx_step, ws.dx_step) + dt_step^2)
        val_norm = sqrt(dot(ws.x_nxt, ws.x_nxt) + t_out^2)

        # Relative convergence check
        if step_norm < rel_tol * val_norm
            @. ws.x_nxt += ws.dx_step
            t_out += dt_step
            return true, ws.x_nxt, t_out
        end

        # Only fail if we exceed limits *after* establishing non-convergence
        if i >= iters
            return false, ws.x_nxt, t_out
        end

        @. ws.x_nxt += ws.dx_step
        t_out += dt_step
        i += 1
    end
end

function validate_game(utils::NTuple{N, AbstractArray{F, N}}) where {F, N}
    if N <= 1
        throw(ArgumentError("A normal-form game must have at least 2 players; got N = $N."))
    end

    if any(isempty, utils)
        throw(ArgumentError("Utility matrices cannot be empty."))
    end

    if !allequal(size, utils)
        throw(DimensionMismatch("All utility matrices must have matching sizes. Received sizes: $(map(size, utils))"))
    end
end

function nash(utils::NTuple{N, AbstractArray{F, N}}; max_iters=1000, end_t=1e6) where {F, N}
    validate_game(utils)

    x = uniform_xprofile(utils)
    t = 0.0
    dx = zero(x)
    dt = 1.0
    ds = 0.01
    iteration = 0
    successes_in_row = 0

    ws = make_hc_workspace(x, utils)

    while t <= end_t && iteration <= max_iters
        dx, dt = predict!(dx, x, t, dx, dt, utils, ws)

        corrected = false
        t_new = t

        while !corrected
            corrected, x_new, t_new = correct!(x, t, dx, dt, ds, utils, ws)

            if !corrected
                ds /= 2
                successes_in_row = 0
                if ds < 1e-4
                    error("Progress along path stalled! Stepsize reduced below 1e-4.")
                end
            else
                copyto!(x, x_new)
                t = t_new

                successes_in_row += 1
                if successes_in_row >= 5
                    successes_in_row = 0
                    ds *= 2
                end

                iteration += 1
            end
        end
    end

    mu = splitviews(x, size(utils[1]) .- 1)
    pi = redlograt_to_prob.(mu)

    return pi
end