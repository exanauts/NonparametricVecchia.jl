module NonparametricVecchiaMadNLPExt

using LinearAlgebra
using KernelAbstractions
using MadNLP
using NonparametricVecchia
using NonparametricVecchia: VecchiaModel, VecchiaKKTSystem

#=
    Structure-exploiting KKT system for the optimization problem

        min  f(x, w)   s.t.   exp(wₖ) - x_{dₖ} = 0,   l ≤ w ≤ u,

    where x ∈ ℝᵖ stores the nonzeros of the factor (columnwise) and w ∈ ℝⁿ stores
    the logarithms of its diagonal entries.

    After the elimination of the bound multipliers (done by MadNLP), the Newton system reads

        [ Hₓ  0   Jᵀ ] [Δx]   [rₓ]
        [ 0   Q   D  ] [Δw] = [r_w]
        [ J   D   Δc ] [Δλ]   [r_λ]

    with Hₓ = H + Σₓ block diagonal (blocks Hⱼ of size mⱼ × mⱼ), Q, D and Δc diagonal,
    and J with a single nonzero per row, located at the diagonal entry x_{dₖ}.
    Eliminating Δx gives n independent 2 × 2 systems

        [ Q   D     ] [Δw]   [r_w             ]
        [ D   Δc - M] [Δλ] = [r_λ - J Hₓ⁻¹ rₓ ],     M = J Hₓ⁻¹ Jᵀ diagonal,

    and Δx = Hₓ⁻¹ (rₓ - Jᵀ Δλ).

    The blocks Hⱼ are factorized with a Cholesky factorization. As long as the Hessian
    of the objective and the primal regularization of x do not change, the factorization
    is reused across the iterations of the interior-point method.
=#

# Position of the coefficient (i, j), i ≥ j, in a block of size m whose lower triangular
# part is stored columnwise. This is the storage used for the Hessian in `VecchiaModel`.
@inline _pidx(i, j, m) = (j - 1) * m - ((j - 1) * (j - 2)) ÷ 2 + (i - j + 1)

# In-place Cholesky factorization of a packed block. Returns 0 on success and 1 if the
# block is not positive definite.
@inline function _packed_cholesky!(L, ho, mj)
    T = eltype(L)
    for c in 1:mj
        pcc = ho + _pidx(c, c, mj)
        s = L[pcc]
        for k in 1:c-1
            lck = L[ho + _pidx(c, k, mj)]
            s -= lck * lck
        end
        if !(s > zero(T))
            return 1
        end
        lcc = sqrt(s)
        L[pcc] = lcc
        for r in c+1:mj
            prc = ho + _pidx(r, c, mj)
            s = L[prc]
            for k in 1:c-1
                s -= L[ho + _pidx(r, k, mj)] * L[ho + _pidx(c, k, mj)]
            end
            L[prc] = s / lcc
        end
    end
    return 0
end

# In-place solve of LLᵀ v = v with a packed Cholesky factor.
@inline function _packed_solve!(v, L, xo, ho, mj)
    for r in 1:mj
        s = v[xo + r]
        for k in 1:r-1
            s -= L[ho + _pidx(r, k, mj)] * v[xo + k]
        end
        v[xo + r] = s / L[ho + _pidx(r, r, mj)]
    end
    for r in mj:-1:1
        s = v[xo + r]
        for k in r+1:mj
            s -= L[ho + _pidx(k, r, mj)] * v[xo + k]
        end
        v[xo + r] = s / L[ho + _pidx(r, r, mj)]
    end
    return
end

# Factorize Hⱼ + Σₓⱼ if it changed since the last factorization, and compute the column
# of (Hⱼ + Σₓⱼ)⁻¹ associated with the diagonal entry of the factor.
@kernel function _vecchia_factorize_kernel!(L, hess_copy, pr_copy, hinv_col, minv, info,
                                            @Const(hess), @Const(pr_diag), @Const(m),
                                            @Const(xoff), @Const(hoff), @Const(dloc))
    j = @index(Global)
    T = eltype(L)
    mj = m[j]
    xo = xoff[j]
    ho = hoff[j]
    nh = (mj * (mj + 1)) ÷ 2

    changed = false
    for k in 1:nh
        changed |= hess[ho + k] != hess_copy[ho + k]
    end
    for k in 1:mj
        changed |= pr_diag[xo + k] != pr_copy[xo + k]
    end

    if changed
        for k in 1:nh
            hess_copy[ho + k] = hess[ho + k]
            L[ho + k] = hess[ho + k]
        end
        for k in 1:mj
            pr_copy[xo + k] = pr_diag[xo + k]
            L[ho + _pidx(k, k, mj)] += pr_diag[xo + k]
        end
        flag = _packed_cholesky!(L, ho, mj)
        info[j] = flag
        if flag == 0
            d = dloc[j]
            for k in 1:mj
                hinv_col[xo + k] = (k == d) ? one(T) : zero(T)
            end
            _packed_solve!(hinv_col, L, xo, ho, mj)
            minv[j] = hinv_col[xo + d]
        end
    end
end

@kernel function _vecchia_solve_kernel!(v, @Const(L), @Const(m), @Const(xoff), @Const(hoff))
    j = @index(Global)
    _packed_solve!(v, L, xoff[j], hoff[j], m[j])
end

# v[block j] -= coef[j] * hinv_col[block j]
@kernel function _vecchia_update_kernel!(v, @Const(hinv_col), @Const(coef), @Const(m), @Const(xoff))
    j = @index(Global)
    xo = xoff[j]
    cj = coef[j]
    for k in 1:m[j]
        v[xo + k] -= cj * hinv_col[xo + k]
    end
end

# y[block j] = beta * y[block j] + alpha * Hⱼ x[block j]
@kernel function _vecchia_mul_kernel!(y, @Const(hess), @Const(x), alpha, beta,
                                      @Const(m), @Const(xoff), @Const(hoff))
    j = @index(Global)
    mj = m[j]
    xo = xoff[j]
    ho = hoff[j]
    for i in 1:mj
        acc = zero(eltype(y))
        for k in 1:mj
            idx = i ≥ k ? _pidx(i, k, mj) : _pidx(k, i, mj)
            acc += hess[ho + idx] * x[xo + k]
        end
        y[xo + i] = iszero(beta) ? alpha * acc : beta * y[xo + i] + alpha * acc
    end
end

"""
    VecchiaBlockSolver

Linear solver associated with `VecchiaKKTSystem`: block Cholesky factorization of the
Hessian of the objective, followed by the solution of `n` independent `2 × 2` systems.
The inertia of the KKT matrix is computed analytically.
"""
mutable struct VecchiaBlockSolver{T, VT, VI} <: MadNLP.AbstractLinearSolver{T}
    p::Int                   # number of nonzeros in the factor
    n::Int                   # number of columns of the factor
    nnzh_obj::Int            # number of nonzeros in the lower triangular part of the blocks Hⱼ
    m::VI                    # size of each block Hⱼ
    xoff::VI                 # offset of the block j in x
    hoff::VI                 # offset of the block j in the packed Hessian
    dloc::VI                 # local position of the diagonal entry in the block j
    diagL::VI                # position of the diagonal entries in x
    # References to the data of the KKT system
    hess::VT
    jac::VT
    pr_diag::VT
    du_diag::VT
    # Factorization of the blocks Hⱼ + Σₓⱼ
    L::VT
    hess_copy::VT
    pr_copy::VT
    hinv_col::VT             # columns (Hⱼ + Σₓⱼ)⁻¹ e_{dⱼ}
    minv::VT                 # diagonal coefficients of (Hⱼ + Σₓⱼ)⁻¹ at the positions dⱼ
    info::VI
    # 2 × 2 systems [q d; d e]
    jx::VT
    jw::VT
    q::VT
    e::VT
    det::VT
    buffer::VT
    inertia::Tuple{Int, Int, Int}
end

struct VecchiaKKT{T, VT, MT, QN, VI} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, QN}
    hess::VT
    jac::VT
    quasi_newton::QN
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    linear_solver::VecchiaBlockSolver{T, VT, VI}
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
end

function _to_device(ref::AbstractVector, v::Vector{Int})
    w = similar(ref, Int, length(v))
    copyto!(w, v)
    return w
end

function MadNLP.create_kkt_system(
    ::Type{VecchiaKKTSystem},
    cb::MadNLP.SparseCallback{T, VT},
    linear_solver::Type;
    hessian_approximation=MadNLP.ExactHessian,
    kwargs...,
) where {T, VT}
    nlp = cb.nlp
    nlp isa VecchiaModel || error("VecchiaKKTSystem only supports a VecchiaModel.")
    hessian_approximation <: MadNLP.ExactHessian || error("VecchiaKKTSystem requires the exact Hessian.")
    isempty(cb.ind_ineq) || error("VecchiaKKTSystem does not support inequality constraints.")
    isempty(cb.ind_fixed) || error("VecchiaKKTSystem does not support fixed variables.")

    cache = nlp.cache
    n = cache.n
    p = cache.nnzL
    nnzh_obj = cache.nnzh_tri_obj
    @assert cb.nvar == p + n
    @assert cb.ncon == n
    @assert cb.nnzj == 2 * n
    @assert cb.nnzh == nnzh_obj + n

    # Block structure of the Hessian of the objective
    m_cpu = Vector{Int}(Array(cache.m))
    xoff_cpu = cumsum([0; m_cpu[1:end-1]])
    hoff_cpu = cumsum([0; (m_cpu .* (m_cpu .+ 1) .÷ 2)[1:end-1]])
    dloc_cpu = Vector{Int}(Array(cache.diagL)) .- xoff_cpu
    @assert all(1 .≤ dloc_cpu .≤ m_cpu)

    ref = cb.ind_lb
    m = _to_device(ref, m_cpu)
    xoff = _to_device(ref, xoff_cpu)
    hoff = _to_device(ref, hoff_cpu)
    dloc = _to_device(ref, dloc_cpu)
    diagL = _to_device(ref, Vector{Int}(Array(cache.diagL)))

    nlb = length(cb.ind_lb)
    nub = length(cb.ind_ub)
    zeros_vt(k) = fill!(VT(undef, k), zero(T))

    hess = zeros_vt(cb.nnzh)
    jac = zeros_vt(cb.nnzj)
    reg = zeros_vt(p + n)
    pr_diag = zeros_vt(p + n)
    du_diag = zeros_vt(n)
    l_diag = zeros_vt(nlb)
    u_diag = zeros_vt(nub)
    l_lower = zeros_vt(nlb)
    u_lower = zeros_vt(nub)

    # NaN ensures that all blocks are factorized the first time.
    hess_copy = fill!(VT(undef, nnzh_obj), T(NaN))
    pr_copy = fill!(VT(undef, p), T(NaN))
    info = _to_device(ref, zeros(Int, n))

    solver = VecchiaBlockSolver{T, VT, typeof(m)}(
        p, n, nnzh_obj, m, xoff, hoff, dloc, diagL,
        hess, jac, pr_diag, du_diag,
        zeros_vt(nnzh_obj), hess_copy, pr_copy, zeros_vt(p), zeros_vt(n), info,
        zeros_vt(n), zeros_vt(n), zeros_vt(n), zeros_vt(n), zeros_vt(n), zeros_vt(n),
        (0, 0, 0),
    )

    quasi_newton = MadNLP.create_quasi_newton(MadNLP.ExactHessian, cb, p + n)
    return VecchiaKKT{T, VT, Matrix{T}, typeof(quasi_newton), typeof(m)}(
        hess, jac, quasi_newton,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        solver,
        cb.ind_ineq, cb.ind_lb, cb.ind_ub,
    )
end

MadNLP.num_variables(kkt::VecchiaKKT) = length(kkt.pr_diag)
MadNLP.get_kkt(kkt::VecchiaKKT) = kkt
MadNLP.get_jacobian(kkt::VecchiaKKT) = kkt.jac
MadNLP.get_hessian(kkt::VecchiaKKT) = kkt.hess
MadNLP.compress_jacobian!(kkt::VecchiaKKT) = nothing
MadNLP.compress_hessian!(kkt::VecchiaKKT) = nothing
MadNLP.build_kkt!(kkt::VecchiaKKT) = nothing
MadNLP.nnz_jacobian(kkt::VecchiaKKT) = length(kkt.jac)

function Base.size(kkt::VecchiaKKT)
    ls = kkt.linear_solver
    N = ls.p + 2 * ls.n
    return (N, N)
end
Base.size(kkt::VecchiaKKT, dim::Int) = size(kkt)[dim]

function MadNLP.initialize!(kkt::VecchiaKKT{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.hess, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    return
end

function MadNLP.jtprod!(y::AbstractVector, kkt::VecchiaKKT{T}, x::AbstractVector) where T
    ls = kkt.linear_solver
    p, n = ls.p, ls.n
    fill!(view(y, 1:p), zero(T))
    view(y, ls.diagL) .= view(kkt.jac, 1:n) .* x
    view(y, p+1:p+n) .= view(kkt.jac, n+1:2n) .* x
    return y
end

function _block_mul!(y, hess, x, alpha, beta, ls::VecchiaBlockSolver)
    backend = KernelAbstractions.get_backend(y)
    _vecchia_mul_kernel!(backend)(y, hess, x, alpha, beta, ls.m, ls.xoff, ls.hoff; ndrange=ls.n)
    KernelAbstractions.synchronize(backend)
    return y
end

function MadNLP.mul_hess_blk!(wx, kkt::VecchiaKKT{T}, t) where T
    ls = kkt.linear_solver
    p, n = ls.p, ls.n
    _block_mul!(wx, kkt.hess, t, one(T), zero(T), ls)
    view(wx, p+1:p+n) .= view(kkt.hess, ls.nnzh_obj+1:ls.nnzh_obj+n) .* view(t, p+1:p+n)
    wx .+= t .* kkt.pr_diag
    return wx
end

function LinearAlgebra.mul!(
    w::MadNLP.AbstractKKTVector{T},
    kkt::VecchiaKKT,
    x::MadNLP.AbstractKKTVector,
    alpha = one(T),
    beta = zero(T),
) where T
    ls = kkt.linear_solver
    p, n = ls.p, ls.n
    wp, xp = MadNLP.primal(w), MadNLP.primal(x)
    wy, xy = MadNLP.dual(w), MadNLP.dual(x)
    jx = view(kkt.jac, 1:n)
    jw = view(kkt.jac, n+1:2n)
    hw = view(kkt.hess, ls.nnzh_obj+1:ls.nnzh_obj+n)
    ww = view(wp, p+1:p+n)
    xw = view(xp, p+1:p+n)

    # Rows associated with x: H xₓ + Jᵀ x_λ
    _block_mul!(wp, kkt.hess, xp, alpha, beta, ls)
    view(wp, ls.diagL) .+= alpha .* jx .* xy
    # Rows associated with w: Λ D x_w + D x_λ, and with λ: J xₓ + D x_w
    if iszero(beta)
        ww .= alpha .* (hw .* xw .+ jw .* xy)
        wy .= alpha .* (jx .* view(xp, ls.diagL) .+ jw .* xw)
    else
        ww .= beta .* ww .+ alpha .* (hw .* xw .+ jw .* xy)
        wy .= beta .* wy .+ alpha .* (jx .* view(xp, ls.diagL) .+ jw .* xw)
    end

    MadNLP._kktmul!(w, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end

# Inertia of the 2 × 2 matrix [q d; d e] with determinant δ.
@inline _npos(δ, q, e) = δ < 0 ? 1 : (δ > 0 ? (q + e > 0 ? 2 : 0) : (q + e > 0 ? 1 : 0))
@inline _nzero(δ, q, e) = δ != 0 ? 0 : (q + e == 0 ? 2 : 1)

function MadNLP.factorize!(ls::VecchiaBlockSolver{T}) where T
    p, n = ls.p, ls.n
    backend = KernelAbstractions.get_backend(ls.L)
    _vecchia_factorize_kernel!(backend)(
        ls.L, ls.hess_copy, ls.pr_copy, ls.hinv_col, ls.minv, ls.info,
        ls.hess, ls.pr_diag, ls.m, ls.xoff, ls.hoff, ls.dloc; ndrange=n,
    )
    KernelAbstractions.synchronize(backend)

    ls.jx .= view(ls.jac, 1:n)
    ls.jw .= view(ls.jac, n+1:2n)
    ls.q .= view(ls.hess, ls.nnzh_obj+1:ls.nnzh_obj+n) .+ view(ls.pr_diag, p+1:p+n)
    ls.e .= ls.du_diag .- ls.jx .^ 2 .* ls.minv
    ls.det .= ls.q .* ls.e .- ls.jw .^ 2

    if any(!iszero, ls.info)
        # One block Hⱼ + Σₓⱼ is not positive definite: report a wrong inertia
        # such that MadNLP increases the primal regularization.
        ls.inertia = (0, 0, p + 2n)
    else
        npos = mapreduce(_npos, +, ls.det, ls.q, ls.e)
        nzero = mapreduce(_nzero, +, ls.det, ls.q, ls.e)
        ls.inertia = (p + npos, nzero, 2n - npos - nzero)
    end
    return ls
end

MadNLP.inertia(ls::VecchiaBlockSolver) = ls.inertia
MadNLP.is_inertia(::VecchiaBlockSolver) = true
MadNLP.improve!(::VecchiaBlockSolver) = false
MadNLP.introduce(::VecchiaBlockSolver) = "VecchiaBlockSolver (block Cholesky)"
MadNLP.is_supported(::Type{<:VecchiaBlockSolver}, ::Type{T}) where T <: AbstractFloat = true

function MadNLP.solve_kkt!(kkt::VecchiaKKT, w::MadNLP.AbstractKKTVector)
    ls = kkt.linear_solver
    p, n = ls.p, ls.n
    backend = KernelAbstractions.get_backend(ls.L)

    MadNLP.reduce_rhs!(kkt, w)
    wp = MadNLP.primal(w)
    ww = view(wp, p+1:p+n)
    wy = MadNLP.dual(w)

    # Δx₀ = Hₓ⁻¹ rₓ
    _vecchia_solve_kernel!(backend)(wp, ls.L, ls.m, ls.xoff, ls.hoff; ndrange=n)
    KernelAbstractions.synchronize(backend)

    # r_λ - J Δx₀
    wy .-= ls.jx .* view(wp, ls.diagL)

    # 2 × 2 systems
    ls.buffer .= (ls.e .* ww .- ls.jw .* wy) ./ ls.det
    wy .= (ls.q .* wy .- ls.jw .* ww) ./ ls.det
    ww .= ls.buffer

    # Δx = Δx₀ - Hₓ⁻¹ Jᵀ Δλ
    ls.buffer .= ls.jx .* wy
    _vecchia_update_kernel!(backend)(wp, ls.hinv_col, ls.buffer, ls.m, ls.xoff; ndrange=n)
    KernelAbstractions.synchronize(backend)

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

end # module
