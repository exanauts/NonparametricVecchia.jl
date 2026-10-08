#=
    Dense block operations used by `VecchiaKKTSystem`.
    The Hessian of the objective is block diagonal, and the lower triangular part of each
    block Hⱼ is stored columnwise (packed storage) in the nonzeros of the Hessian.
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
@inline function _vecchia_factorize_block!(j, L, hess_copy, pr_copy, hinv_col, minv, info,
                                           hess, pr_diag, m, xoff, hoff, dloc)
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
    return
end

# v[block j] -= coef[j] * hinv_col[block j]
@inline function _vecchia_update_block!(j, v, hinv_col, coef, m, xoff)
    xo = xoff[j]
    cj = coef[j]
    for k in 1:m[j]
        v[xo + k] -= cj * hinv_col[xo + k]
    end
    return
end

# y[block j] = beta * y[block j] + alpha * Hⱼ x[block j]
@inline function _vecchia_mul_block!(j, y, hess, x, alpha, beta, m, xoff, hoff)
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
    return
end

#=
    Operations on all blocks, used by `VecchiaKKTSystem`.
    The methods below run on CPU; the CUDA extension implements them with GPU kernels.
=#
function vecchia_factorize_blocks!(L::Vector, hess_copy, pr_copy, hinv_col, minv, info,
                                   hess, pr_diag, m, xoff, hoff, dloc)
    for j in eachindex(m)
        _vecchia_factorize_block!(j, L, hess_copy, pr_copy, hinv_col, minv, info,
                                  hess, pr_diag, m, xoff, hoff, dloc)
    end
    return L
end

function vecchia_solve_blocks!(v, L::Vector, m, xoff, hoff)
    for j in eachindex(m)
        _packed_solve!(v, L, xoff[j], hoff[j], m[j])
    end
    return v
end

function vecchia_update_blocks!(v, hinv_col::Vector, coef, m, xoff)
    for j in eachindex(m)
        _vecchia_update_block!(j, v, hinv_col, coef, m, xoff)
    end
    return v
end

function vecchia_mul_blocks!(y, hess::Vector, x, alpha, beta, m, xoff, hoff)
    for j in eachindex(m)
        _vecchia_mul_block!(j, y, hess, x, alpha, beta, m, xoff, hoff)
    end
    return y
end
