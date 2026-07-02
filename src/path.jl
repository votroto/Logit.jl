
function givens(b::Matrix{Float64}, q::Matrix{Float64}, c1::Ref{Float64}, c2::Ref{Float64}, l1::Int, l2::Int, l3::Int)
    if abs(c1) + abs(c2) == 0.0
        return
    end

    if (abs(c2) >= abs(c1))
        sn = std::sqrt(1.0 + sqr(c1 / c2)) * abs(c2)
    else
        sn = std::sqrt(1.0 + sqr(c2 / c1)) * abs(c1)
    end
    s1::Float64 = c1 / sn;
    s2::Float64 = c2 / sn;

    for k in 1:size(q, 2)
        sv1::Float64 = q[l1, k];
        sv2::Float64 = q[l2, k];
        q[l1, k] = s1 * sv1 + s2 * sv2;
        q[l2, k] = -s2 * sv1 + s1 * sv2;
    end

    for k in l3:size(b, 2)
        sv1::Float64 = b[l1, k];
        sv2::Float64 = b[l2, k];
        b[l1, k] = s1 * sv1 + s2 * sv2;
        b[l2, k] = -s2 * sv1 + s1 * sv2;
    end

    c1 = sn
    c2 = 0.0
end

function set_as_identity(M::Matrix{Float64})
    M = fill!(out, 0.0);
    for i in 0:size(M, 1)
        M[i, i] = 1.0;
    end
end


function QR_decomp(b::Matrix{Float64}, q::Matrix{Float64})
    set_as_identity(q);
    for m in 1:size(b, 2), k in (m+1):size(b, 1)
        givens(b, q, b[m, m], b[k, m], m, k, m + 1);
    end
end


function newton_step(q::Matrix{Float64}, b::Matrix{Float64}, u::Vector{Float64}, y::Vector{Float64}, d::Ref{Float64})
    for k in 1:size(b, 2)
        for l in 1:(k-1)
            y[k] -= b(l, k) * y[l];
        end
        y[k] /= b(k, k);
    end

    d = 0.0;
    for k in 1:size(b, 1)
        s = 0.0;
        for l in 1:size(b, 2)
            s += q(l, k) * y[l];
        end
        u[k] -= s;
        d += s * s;
    end
    d = sqrt(d);
end

function trace_path(p_function::Function, p_jacobian::Function, x::Vector{Float64}, p_omega::Ref{Float64}, p_terminate, p_callback, m_hStart, m_maxDecel)

    c_tol::Float64 = 1.0e-4;   # tolerance for corrector iteration
    c_maxDist::Float64 = 0.4;  # maximal distance to curve
    c_maxContr::Float64 = 0.6; # maximal contraction rate in corrector
    c_eta::Float64 = 0.1;      # perturbation to avoid cancellation
    # in calculating contraction rate
    h::Float64 = m_hStart;           # initial stepsize
    c_hmin::Float64 = 1.0e-8;  # minimal stepsize
    c_maxIter::Int = 100;     # maximum iterations in corrector

    c_pert::Float64 = 0.0000001; # The size of perturbation to apply to avoid bifurcation traps
    pert::Float64 = 0.0;               # The current version of the perturbation being applied
    pert_countdown::Float64 = 0.0;     # How much longer (in arclength) to apply perturbation

    u = Vector{Float64}(undef, x.size());
    t = Vector{Float64}(undef, x.size());
    newT = Vector{Float64}(undef, x.size());# t is current tangent at x; newT is tangent at u, which is the next point.
    y = Vector{Float64}(undef, x.size() - 1);
    b = Matrix{Float64}(undef, x.size(), x.size() - 1);
    q = Matrix{Float64}(undef, x.size(), x.size());

    p_jacobian(x, b);
    QRDecomp(b, q);
    t .= q[end, :]
    p_callback(x);

    while (!p_terminate(x))
        accept = true;

        if abs(h) <= c_hmin
            return;
        end

        # Predictor step
        for k = 1:length(x)
            u[k] = x[k] + h * p_omega * t[k];
        end
        decel::Float64 = 1.0 / m_maxDecel; # initialize deceleration factor
        p_jacobian(u, b);
        QRDecomp(b, q);

        iter::Int = 1;
        disto::Float64 = 0.0;
        while (true)
            dist::Float64;

            p_function(u, y);
            y[1] += pert;
            NewtonStep(q, b, u, y, dist);

            if (dist >= c_maxDist)
                accept = false;
                break;
            end

            decel = max(decel, sqrt(dist / c_maxDist) * m_maxDecel);
            if (iter >= 2)
                contr::Float64 = dist / (disto + c_tol * c_eta);
                if (contr > c_maxContr)
                    accept = false;
                    break;
                end
                decel = max(decel, sqrt(contr / c_maxContr) * m_maxDecel);
            end

            if (dist <= c_tol)
                # Success; break out of iteration
                break;
            end
            disto = dist;
            iter += 1;
            if (iter > c_maxIter)
                return;
            end
        end

        # Obtain the tangent at the next step
        newT .= q[end, :]
        omega_flip::Float64 = (t * newT < 0.0) ? -1.0 : 1.0;

        if (omega_flip == -1.0)
            # The orientation of the curve has changed, indicating a bifurcation.
            # Switch on perturbation and attempt to continue following the branch that
            # is oriented in the same direction as we were originally following
            if (pert_countdown == 0.0)
                pert = c_pert;
                pert_countdown = abs(2 * h);
            end
            accept = false;
        end

        if (!accept)
            h /= m_maxDecel; # PC not accepted; change stepsize and retry
            if (abs(h) <= c_hmin)
                return;
            end
            continue;
        end

        # Determine new stepsize
        decel = min(decel, m_maxDecel);


            h = abs(h / decel);

        # PC step was successful; update and iterate
        x = u;
        t = newT;
        p_callback(x);

        if (pert_countdown > 0.0)
            # If we are currently perturbing in the neighborhood of a bifurcation, check to see
            # whether we think we are likely past it, and switch off if we are.
            pert_countdown -= abs(h);
            if (pert_countdown < 0.0)
                pert = 0.0;
                pert_countdown = 0.0;
            end
        end
    end
end