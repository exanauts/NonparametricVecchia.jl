module NonparametricVecchiaCUDAExt

using LinearAlgebra
using NLPModels
using NonparametricVecchia
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using SparseArrays

function NonparametricVecchia.VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::CuMatrix{T};
                                           lvar_diag::Union{Nothing,CuVector{T}}=nothing, 
                                           uvar_diag::Union{Nothing,CuVector{T}}=nothing,
                                           lambda::Real=0, format::Symbol=:coo, uplo::Symbol=:L) where T
    S = CuArray{T, 1, CUDA.DeviceMemory}
    cache = NonparametricVecchia.create_vecchia_cache(I, J, samples, T(lambda), format, uplo)

    nvar = length(cache.rowsL) + length(cache.colptrL) - 1
    ncon = length(cache.colptrL) - 1

    # Allocating data
    x0 = fill!(S(undef, nvar), zero(T))
    y0 = fill!(S(undef, ncon), zero(T))
    lcon = fill!(S(undef, ncon), zero(T))
    ucon = fill!(S(undef, ncon), zero(T))
    lvar = fill!(S(undef, nvar), -Inf)
    uvar = fill!(S(undef, nvar),  Inf)

    # Apply box constraints to the diagonal of L through the variables w = log(diag(L)).
    # Bounding w instead of the diagonal entries of L keeps the Hessian block of L
    # independent of the iterate, which is exploited by `VecchiaKKTSystem`.
    w_lvar = view(lvar, cache.nnzL+1:nvar)
    w_uvar = view(uvar, cache.nnzL+1:nvar)
    if !isnothing(lvar_diag)
        w_lvar .= log.(lvar_diag)
    else
        w_lvar .= log(T(1e-10))
    end

    if !isnothing(uvar_diag)
        w_uvar .= log.(uvar_diag)
    else
        w_uvar .= log(T(1e10))
    end

    view(x0, cache.diagL) .= 1.0

    meta = NLPModelMeta{T, S}(
        nvar,
        ncon = ncon,
        x0 = x0,
        name = "nonparametric_vecchia_gpu",
        nnzj = 2*cache.n,
        nnzh = cache.nnzh_tri_lag,
        y0 = y0,
        lcon = lcon,
        ucon = ucon,
        lvar = lvar,
        uvar = uvar,
        minimize=true,
        islp=false,
        lin_nnzj = 0
    )
    
    return VecchiaModel(meta, Counters(), cache)
end


function NonparametricVecchia.create_vecchia_cache(I::Vector{Int}, J::Vector{Int}, samples::CuMatrix{T},
                                                   lambda::T, format::Symbol, uplo::Symbol) where {T}
    S = CuArray{T, 1, CUDA.DeviceMemory}
    Msamples, n = size(samples)

    if format == :coo
        nnz_coo = length(I)
        V = ones(Int, nnz_coo)
        P = sparse(I, J, V, n, n)

        # SPARSITY PATTERN OF L IN CSC FORMAT.
        rowsL = P.rowval
        colptrL = P.colptr
    elseif format == :csc
        rowsL = I
        colptrL = J
    else
        error("Unsupported format = $format for the sparsity pattern.")
    end

    nnzL = length(rowsL)
    m = [colptrL[j+1] - colptrL[j] for j in 1:n]

    # Number of nonzeros in the the lower triangular part of the Hessians
    nnzh_tri_obj = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag = nnzh_tri_obj + n

    offsets = cumsum([0; m[1:end-1]]) |> CuVector{Int}
    B = [CuMatrix{T}(undef, 0, 0)]

    rowsL = CuVector{Int}(rowsL)
    colptrL = CuVector{Int}(colptrL)
    m = CuVector{Int}(m)

    hess_obj_vals = S(undef, nnzh_tri_obj)
    NonparametricVecchia.vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m)

    if uplo == :L
        diagL = colptrL[1:n]
    elseif uplo == :U
        diagL = colptrL[2:n+1]
        diagL .-= 1
    else
        error("Unsupported uplo = $uplo")
    end
    buffer = S(undef, nnzL)

    return NonparametricVecchia.VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, diagL,
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

function NonparametricVecchia.recover_factor(nlp::VecchiaModel{T,<:CuVector{T}}, solution::CuVector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = CuSparseMatrixCSC(colptr, rowval, nzval, (n, n))
    return factor
end

function NonparametricVecchia.vecchia_mul!(y::CuVector{T}, B::Vector{<:CuMatrix{T}}, hess_obj_vals::CuVector{T},
                                           x::CuVector{T}, n::Int, m::CuVector{Int}, offsets::CuVector{Int}) where T <: AbstractFloat
    # Reset the vector y
    fill!(y, zero(T))

    # Launch the kernel
    backend = KernelAbstractions.get_backend(y)
    kernel = vecchia_mul_kernel!(backend)
    kernel(y, hess_obj_vals, x, m, offsets, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return y
end

@kernel function vecchia_mul_kernel!(y, @Const(hess_obj_vals), @Const(x), @Const(m), @Const(offsets))
    index = @index(Global)
    offset = offsets[index]
    mj = m[index]
    pos = 0
    for i = 1:index-1
        pos += m[i] * (m[i] + 1) ÷ 2
    end

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

function NonparametricVecchia.vecchia_build_B!(B::Vector{<:CuMatrix{T}}, samples::CuMatrix{T}, lambda::T,
                                               rowsL::CuVector{Int}, colptrL::CuVector{Int}, hess_obj_vals::CuVector{T},
                                               n::Int, m::CuVector{Int}) where T <: AbstractFloat
    # Launch the kernel
    backend = KernelAbstractions.get_backend(samples)
    r = size(samples, 1)
    kernel = vecchia_build_B_kernel!(backend)
    kernel(hess_obj_vals, samples, lambda, rowsL, colptrL, m, r, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function vecchia_build_B_kernel!(hess_obj_vals, @Const(samples), @Const(lambda), @Const(rowsL), @Const(colptrL), @Const(m), @Const(r))
    index = @index(Global)
    col = colptrL[index]
    mj = m[index]

    pos = 0
    for i = 1:index-1
        pos += m[i] * (m[i] + 1) ÷ 2
    end

    for s in 1:mj
        for t in s:mj
            if s ≤ t
                pos = pos + 1
                acc = 0.0
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

function NonparametricVecchia.vecchia_generate_hess_tri_structure!(n::Int, m::CuVector{Int}, nnzL::Int, nnzh_tri_obj::Int,
                                                                   offsets::CuVector{Int}, hrows::CuVector{Int}, hcols::CuVector{Int})
    # launch the kernel
    backend = KernelAbstractions.get_backend(hrows)
    kernel = vecchia_generate_hess_tri_structure_kernel!(backend)
    kernel(n, m, nnzL, nnzh_tri_obj, offsets, hrows, hcols, ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function vecchia_generate_hess_tri_structure_kernel!(@Const(n), @Const(m), @Const(nnzL), @Const(nnzh_tri_obj), @Const(offsets), hrows, hcols)
    index = @index(Global)
    mj = m[index]
    offset = offsets[index]

    pos = 0
    for i = 1:index-1
        pos += m[i] * (m[i] + 1) ÷ 2
    end

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

end  # end module
