using LinearAlgebra


# ============================================================
# Strategy encoding
# ============================================================

function exp_strategy(ell, player_indices)

    π = similar(ell)

    for r in player_indices
        @views π[r] .= exp.(ell[r])
    end

    return π
end



# ============================================================
# Expected utilities
# ============================================================

function compute_expected_utilities(
    π,
    payoffs,
    player_indices
)

    U = zeros(eltype(π), length(π))

    if length(player_indices)==2

        r1, r2 = player_indices

        mul!(
            view(U, r1),
            payoffs[1],
            view(π, r2)
        )

        mul!(
            view(U, r2),
            payoffs[2]',
            view(π, r1)
        )

    end

    return U
end



# ============================================================
# Utility gradients
# ============================================================

function compute_utility_gradients(
    π,
    payoffs,
    player_indices
)

    dU = [
        zeros(length(r), length(π))
        for r in player_indices
    ]


    if length(player_indices)==2

        r1, r2 = player_indices


        dU[1][:, r2] .= payoffs[1]

        dU[2][:, r1] .= payoffs[2]'

    end


    return dU
end



# ============================================================
# Turocy system in log coordinates
#
# variables are ell = log(pi)
#
# ============================================================

function turocy_system_log!(
    F,
    ell,
    λ,
    payoffs,
    player_indices
)


    π = exp_strategy(
        ell,
        player_indices
    )


    U =
        compute_expected_utilities(
            π,
            payoffs,
            player_indices
        )



    for (i, range) in enumerate(player_indices)


        ref = first(range)


        # normalization:
        #
        # sum exp(ell)=1
        #

        F[ref] =
            sum(exp.(view(ell, range))) - 1.0



        # logit equilibrium equations

        for idx in range[2:end]

            F[idx] =
                ell[idx]
            -
            ell[ref]
            -
            λ*(U[idx]-U[ref])

        end

    end


    return F
end



# ============================================================
# Jacobian wrt ell
#
# ============================================================

function turocy_jacobian_log!(
    J,
    ell,
    λ,
    payoffs,
    player_indices
)


    fill!(J, 0.0)



    π =
        exp_strategy(
            ell,
            player_indices
        )


    dU =
        compute_utility_gradients(
            π,
            payoffs,
            player_indices
        )



    for (i, range) in enumerate(player_indices)


        ref = first(range)



        # ----------------------------------------
        # normalization row
        #
        # d(sum exp(ell))/dell = exp(ell)=pi
        # ----------------------------------------

        for k in range

            J[ref, k]=π[k]

        end



        # ----------------------------------------
        # QRE rows
        # ----------------------------------------

        for idx in range[2:end]


            # derivative of ell[idx]-ell[ref]

            J[idx, idx]+=1.0

            J[idx, ref]-=1.0



            # derivative through utilities

            for k in 1:length(ell)


                du =
                    dU[i][idx-first(range)+1, k]
                -
                dU[i][1, k]


                # chain rule:
                #
                # dpi/dell = pi
                #

                J[idx, k] -=
                    λ * du * π[k]

            end

        end

    end


    return J
end



# ============================================================
# derivative wrt lambda
# ============================================================

function turocy_partial_lambda_log!(
    dF_dλ,
    ell,
    λ,
    payoffs,
    player_indices
)


    π =
        exp_strategy(
            ell,
            player_indices
        )


    U =
        compute_expected_utilities(
            π,
            payoffs,
            player_indices
        )


    for (i, range) in enumerate(player_indices)

        ref=first(range)


        dF_dλ[ref]=0.0


        for idx in range[2:end]

            dF_dλ[idx] =
                -(U[idx]-U[ref])

        end

    end


    return dF_dλ
end



# ============================================================
# Example game
# ============================================================


player_indices =
    [
        1:3,
        4:6
    ]


payoffs_p1 =
    [
        0.0 1.0 4.0;
        1.0 0.0 1.0;
        4.0 1.0 0.0
    ]


payoffs_p2 =
    -
payoffs_p1


payoffs =
    (
        payoffs_p1,
        payoffs_p2
    )



# ============================================================
# Continuation closures
# ============================================================


sys_closure!(
    F, x, t
) =
    turocy_system_log!(
        F,
        x,
        t,
        payoffs,
        player_indices
    )



jac_closure!(
    J, x, t
) =
    turocy_jacobian_log!(
        J,
        x,
        t,
        payoffs,
        player_indices
    )



param_closure!(
    dF, x, t
) =
    turocy_partial_lambda_log!(
        dF,
        x,
        t,
        payoffs,
        player_indices
    )



# ============================================================
# Initial condition
# lambda=0 centroid
#
# ell = log(pi)
# ============================================================


guess =
    log.([
        1/3, 1/3, 1/3,
        1/3, 1/3, 1/3
    ])



# ============================================================
# Run homotopy
# ============================================================


ell_qre =
    hc_turocy(
        guess,
        0.0,
        1000000.0,
        sys_closure!,
        jac_closure!,
        param_closure!
    )



# Convert back to probabilities

π_qre =
    exp_strategy(
        ell_qre,
        player_indices
    )


println("QRE probabilities:")
println(π_qre)