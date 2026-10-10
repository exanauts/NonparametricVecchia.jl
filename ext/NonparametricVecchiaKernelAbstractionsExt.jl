module NonparametricVecchiaKernelAbstractionsExt

using KernelAbstractions
using NonparametricVecchia

#=
    Kernels for the models whose data are stored on a GPU (or any array type that is not
    a `Vector`). The methods for `Vector` are implemented in the main package.
=#
function NonparametricVecchia.vecchia_mul!(y::AbstractVector{T}, B::Vector{<:AbstractMatrix{T}}, hess_obj_vals::AbstractVector{T},
                                           x::AbstractVector{T}, n::Int, m::AbstractVector{Int}, offsets::AbstractVector{Int},
                                           hoffsets::AbstractVector{Int}) where T <: AbstractFloat
    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KernelAbstractions.get_backend(y)
    kernel = vecchia_mul_kernel!(backend)
    kernel(y, hess_obj_vals, x, m, offsets, hoffsets, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return y
end

@kernel function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(offsets), @Const(hoffsets))
    index = @index(Global)
    offset = offsets[index]
    mj = m[index]
    pos = hoffsets[index]

    # Perform the matrix-vector multiplication for the current symmetric block
    for j in 1:mj
        idx1 = (j - 1) * (mj + 1) - j * (j-1) ÷ 2
        for i in j:mj
            idx2 = idx1 + (i - j + 1)
            val = hess_obj_vals[pos+idx2]

            # Diagonal element contributes only once
            if i == j
                y[offset+i] += val * x[offset+j]
            else
                y[offset+i] += val * x[offset+j]
                y[offset+j] += val * x[offset+i]  # due to symmetry A[i,j] = A[j,i]
            end
        end
    end
    nothing
end

function NonparametricVecchia.vecchia_build_B!(B::Vector{<:AbstractMatrix{T}}, samples::AbstractMatrix{T}, lambda::T,
                                               rowsL::AbstractVector{Int}, colptrL::AbstractVector{Int}, hess_obj_vals::AbstractVector{T},
                                               n::Int, m::AbstractVector{Int}, hoffsets::AbstractVector{Int}) where T <: AbstractFloat
    # Launch the kernel
    backend = KernelAbstractions.get_backend(samples)
    r = size(samples, 1)
    kernel = vecchia_build_B_kernel!(backend)
    kernel(hess_obj_vals, samples, lambda, rowsL, colptrL, m, hoffsets, r, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function vecchia_build_B_kernel!(hess_obj_vals, @Const(samples), @Const(lambda), @Const(rowsL), @Const(colptrL), @Const(m), @Const(hoffsets), @Const(r))
    index = @index(Global)
    col = colptrL[index]
    mj = m[index]
    pos = hoffsets[index]

    for s in 1:mj
        for t in s:mj
            if s ≤ t
                pos = pos + 1
                acc = zero(eltype(hess_obj_vals))
                for i = 1:r
                    acc += samples[i, rowsL[col+t-1]] * samples[i, rowsL[col+s-1]]
                end
                if (lambda != 0) && (s == t)
                    acc += lambda
                end
                hess_obj_vals[pos] = acc
            end
        end
    end
    nothing
end

function NonparametricVecchia.vecchia_generate_hess_tri_structure!(n::Int, m::AbstractVector{Int}, nnzL::Int, nnzh_tri_obj::Int,
                                                                   offsets::AbstractVector{Int}, hoffsets::AbstractVector{Int},
                                                                   hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer})
    # launch the kernel
    backend = KernelAbstractions.get_backend(hrows)
    kernel = vecchia_generate_hess_tri_structure_kernel!(backend)
    kernel(n, m, nnzL, nnzh_tri_obj, offsets, hoffsets, hrows, hcols, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function vecchia_generate_hess_tri_structure_kernel!(@Const(n), @Const(m), @Const(nnzL), @Const(nnzh_tri_obj), @Const(offsets), @Const(hoffsets), hrows, hcols)
    index = @index(Global)
    mj = m[index]
    offset = offsets[index]
    pos = hoffsets[index]

    for s in 1:mj
        for t in 1:mj
            if s ≤ t
                pos = pos + 1
                hrows[pos] = offset + t
                hcols[pos] = offset + s
            end
        end
    end

    hrows[nnzh_tri_obj + index] = nnzL + index
    hcols[nnzh_tri_obj + index] = nnzL + index
    nothing
end

#=
    Dense block operations used by `VecchiaKKTSystem`, one GPU thread per block.
=#
@kernel function vecchia_factorize_blocks_kernel!(L, hess_copy, pr_copy, hinv_col, minv, info, nfact,
                                                  @Const(hess), @Const(pr_diag), @Const(m),
                                                  @Const(xoff), @Const(hoff), @Const(dloc))
    j = @index(Global)
    NonparametricVecchia._vecchia_factorize_block!(j, L, hess_copy, pr_copy, hinv_col, minv, info, nfact,
                                                   hess, pr_diag, m, xoff, hoff, dloc)
end

@kernel function vecchia_solve_blocks_kernel!(v, @Const(L), @Const(m), @Const(xoff), @Const(hoff))
    j = @index(Global)
    NonparametricVecchia._packed_solve!(v, L, xoff[j], hoff[j], m[j])
end

@kernel function vecchia_update_blocks_kernel!(v, @Const(hinv_col), @Const(coef), @Const(m), @Const(xoff))
    j = @index(Global)
    NonparametricVecchia._vecchia_update_block!(j, v, hinv_col, coef, m, xoff)
end

@kernel function vecchia_mul_blocks_kernel!(y, @Const(hess), @Const(x), alpha, beta,
                                            @Const(m), @Const(xoff), @Const(hoff))
    j = @index(Global)
    NonparametricVecchia._vecchia_mul_block!(j, y, hess, x, alpha, beta, m, xoff, hoff)
end

function NonparametricVecchia.vecchia_factorize_blocks!(L::AbstractVector, hess_copy, pr_copy, hinv_col, minv, info, nfact,
                                                        hess, pr_diag, m, xoff, hoff, dloc)
    backend = KernelAbstractions.get_backend(L)
    kernel = vecchia_factorize_blocks_kernel!(backend)
    kernel(L, hess_copy, pr_copy, hinv_col, minv, info, nfact, hess, pr_diag, m, xoff, hoff, dloc, ndrange=length(m))
    KernelAbstractions.synchronize(backend)
    return L
end

function NonparametricVecchia.vecchia_solve_blocks!(v, L::AbstractVector, m, xoff, hoff)
    backend = KernelAbstractions.get_backend(L)
    kernel = vecchia_solve_blocks_kernel!(backend)
    kernel(v, L, m, xoff, hoff, ndrange=length(m))
    KernelAbstractions.synchronize(backend)
    return v
end

function NonparametricVecchia.vecchia_update_blocks!(v, hinv_col::AbstractVector, coef, m, xoff)
    backend = KernelAbstractions.get_backend(hinv_col)
    kernel = vecchia_update_blocks_kernel!(backend)
    kernel(v, hinv_col, coef, m, xoff, ndrange=length(m))
    KernelAbstractions.synchronize(backend)
    return v
end

function NonparametricVecchia.vecchia_mul_blocks!(y, hess::AbstractVector, x, alpha, beta, m, xoff, hoff)
    backend = KernelAbstractions.get_backend(hess)
    kernel = vecchia_mul_blocks_kernel!(backend)
    kernel(y, hess, x, alpha, beta, m, xoff, hoff, ndrange=length(m))
    KernelAbstractions.synchronize(backend)
    return y
end

end  # end module
