"""
Internal struct from which to fetch persisting objects in the optimization function.
There is no need for a user to mess with this!

## Fields
- `n` : Size of the problem
- `M` : Number of Samples
- `nnzL` : Number of nonzeros in L
- `colptrL` : Array which indexes the beginning of each column in L
- `rowsL` : Row index of nonzero entries in L
- `diagL` : Position of the diagonal coefficient of L
- `m` : Number of nonzeros in each column of L
- `offsets` : Number of nonzeros of L before the column j, i.e., offset of the block Bⱼ in x
- `hoffsets` : Number of nonzeros in hess_obj_vals before the block Bⱼ
- `B` : Vector of matrices Bⱼ, the constant blocks in the Hessian
- `nnzh_tri_obj` : Number of nonzeros in the lower triangular part of the Hessian of the objective
- `nnzh_tri_lag` : Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
- `hess_obj_vals` : Nonzeros of the lower triangular part of the Hessian of the objective
- `buffer` : Additional buffer needed for the objective

"""
struct VecchiaCache{T, S, VI, M}
    n::Int                              # Size of the problem
    M::Int                              # Number of Samples
    nnzL::Int                           # Number of nonzeros in L
    colptrL::VI                         # Array which indexes the beginning of each column in L
    rowsL::VI                           # Row index of nonzero entries in L
    diagL::VI                           # Position of the diagonal coefficient of L
    m::VI                               # Number of nonzeros in each column of L
    offsets::VI                         # Number of nonzeros of L before the column j, i.e., offset of the block Bⱼ in x
    hoffsets::VI                        # Number of nonzeros in hess_obj_vals before the block Bⱼ
    B::Vector{M}                        # Vector of matrices Bⱼ, the constant blocks in the Hessian
    nnzh_tri_obj::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the objective
    nnzh_tri_lag::Int                   # Number of nonzeros in the lower triangular part of the Hessian of the Lagrangian
    hess_obj_vals::S                    # Nonzeros of the lower triangular part of the Hessian of the objective
    buffer::S                           # Additional buffer needed for the objective
end

mutable struct VecchiaModel{T, S, VI, M} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    cache::VecchiaCache{T, S, VI, M}
end

"""
    nlp = VecchiaModel(L::LowerTriangular, samples; lvar_diag, uvar_diag, lambda)
    nlp = VecchiaModel(U::UpperTriangular, samples; lvar_diag, uvar_diag, lambda)
    nlp = VecchiaModel(I, J, samples; lvar_diag, uvar_diag, lambda, format=:coo, uplo=:L)

Optimization problem whose solution is a sparse triangular factor `L` such that
`L * L'` approximates the inverse of the covariance matrix of the `samples`.
The model is an `AbstractNLPModel` and can be solved with any solver of the
JuliaSmoothOptimizers ecosystem, for instance with MadNLP and
`kkt_system=VecchiaKKTSystem`.

The variables are the nonzeros of the factor, stored columnwise, followed by the
logarithms `w` of its diagonal entries, which are linked to the diagonal entries
by the equality constraints `exp(wⱼ) - Lⱼⱼ = 0`.
Use [`recover_factor`](@ref) to build the factor from a solution.

#### Arguments

- `L` / `U`: sparse lower or upper triangular matrix whose sparsity pattern is the one of the factor;
- `I`, `J`: sparsity pattern of the factor, in COO format (`format=:coo`, row and column indices)
  or CSC format (`format=:csc`, `rowval` and `colptr`), with `uplo=:L` or `uplo=:U`;
- `samples`: matrix of size `m × n`, with one replicate per row. A `CuMatrix` or a `ROCMatrix` builds a model on an NVIDIA or AMD GPU.

#### Keyword arguments

- `lvar_diag`, `uvar_diag`: vectors of length `n` with the lower and upper bounds on the diagonal
  entries of the factor (by default, all bounds are `1e-10` and `1e10`). They are imposed on the
  variables `w = log(diag(L))` as `log.(lvar_diag) ≤ w ≤ log.(uvar_diag)`, so the bounds must be
  nonnegative, and a lower bound equal to `0` means no lower bound. For a model on GPU, the
  vectors can be stored on the CPU or on the GPU;
- `lambda`: ridge penalty `(lambda / 2) * ‖L‖²_F` added to the objective (by default `0`).
  It penalizes all nonzeros of the factor, including its diagonal, and adds `lambda` to the
  diagonal of each block of the Hessian, which makes the blocks positive definite when the
  number of replicates is smaller than the number of nonzeros in a column of the factor.
"""
function VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::Matrix{T};
                      lvar_diag::Union{Nothing,Vector{T}}=nothing, 
                      uvar_diag::Union{Nothing,Vector{T}}=nothing,
                      lambda::Real=0.0, format::Symbol=:coo, uplo::Symbol=:L) where T
    S = Vector{T}
    cache = create_vecchia_cache(I, J, samples, T(lambda), format, uplo)

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
        name = "nonparametric_vecchia_cpu",
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

function VecchiaModel(L::LowerTriangular{G, SparseMatrixCSC{G,Int64}}, samples; 
                      lvar_diag=nothing, uvar_diag=nothing, lambda::Real=0.0) where {G}
  parent = L.data
  istril(parent) || throw(error("Your backing array in the LowerTriangular isn't actually lower triangular. Please call with, e.g., LowerTriangular(tril(L))."))
  VecchiaModel(parent.rowval, parent.colptr, samples; lvar_diag, 
               uvar_diag, lambda, uplo=:L, format=:csc)
end

function VecchiaModel(U::UpperTriangular{G, SparseMatrixCSC{G,Int64}}, samples; 
                      lvar_diag=nothing, uvar_diag=nothing, lambda::Real=0.0) where {G}
  parent = U.data
  istriu(parent) || throw(error("Your backing array in the UpperTriangular isn't actually upper triangular. Please call with, e.g., UpperTriangular(triu(U))."))
  VecchiaModel(parent.rowval, parent.colptr, samples; lvar_diag, 
               uvar_diag, lambda, uplo=:U, format=:csc)
end

# Offsets of the blocks Bⱼ in the vector of variables (`offsets`) and in the nonzeros of
# the lower triangular part of the Hessian (`hoffsets`), computed once so that the GPU
# kernels can process the blocks independently.
function vecchia_block_offsets(m::Vector{Int})
    n = length(m)
    offsets = zeros(Int, n)
    hoffsets = zeros(Int, n)
    for j in 2:n
        offsets[j] = offsets[j-1] + m[j-1]
        hoffsets[j] = hoffsets[j-1] + m[j-1] * (m[j-1] + 1) ÷ 2
    end
    return offsets, hoffsets
end

function create_vecchia_cache(I::Vector{Int}, J::Vector{Int}, samples::Matrix{T},
                              lambda::T, format::Symbol, uplo::Symbol) where {T}
    S = Vector{T}
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

    # Number of nonzeros in the lower triangular part of the Hessians
    nnzh_tri_obj = sum(m[j] * (m[j] + 1) for j in 1:n) ÷ 2
    nnzh_tri_lag = nnzh_tri_obj + n

    offsets, hoffsets = vecchia_block_offsets(m)
    B = [Matrix{T}(undef, m[j], m[j]) for j = 1:n]

    hess_obj_vals = S(undef, nnzh_tri_obj)
    vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m, hoffsets)

    if uplo == :L
        diagL = colptrL[1:n]
    elseif uplo == :U
        diagL = colptrL[2:n+1]
        diagL .-= 1
    else
        error("Unsupported uplo = $uplo")
    end
    buffer = S(undef, nnzL)

    return VecchiaCache{eltype(S), S, typeof(rowsL), typeof(B[1])}(
        n, Msamples, nnzL,
        colptrL, rowsL, diagL,
        m, offsets, hoffsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end

"""
    factor = recover_factor(nlp::VecchiaModel, solution)

Build the sparse triangular factor from a `solution` of `nlp`, i.e., the first
`nlp.cache.nnzL` entries of `solution` stored in the sparsity pattern of the factor.
The factor is a `SparseMatrixCSC` on CPU, a `CuSparseMatrixCSC` on NVIDIA GPUs
(CUDA.jl), and a `ROCSparseMatrixCSC` on AMD GPUs (AMDGPU.jl).
"""
function recover_factor(nlp::VecchiaModel{T,Vector{T}}, solution::Vector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = SparseMatrixCSC(n, n, colptr, rowval, nzval)
    return factor
end
