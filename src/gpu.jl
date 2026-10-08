#=
    Construction of a `VecchiaModel` whose data are stored on a GPU.
    The GPU extensions (CUDA, AMDGPU) only specify the array types; the computations are
    performed by the kernels of the KernelAbstractions extension.
=#
function vecchia_model_gpu(I::Vector{Int}, J::Vector{Int}, samples::AbstractMatrix{T}, name::String;
                           lvar_diag::Union{Nothing,AbstractVector{T}}=nothing,
                           uvar_diag::Union{Nothing,AbstractVector{T}}=nothing,
                           lambda::Real=0, format::Symbol=:coo, uplo::Symbol=:L) where T
    S = typeof(similar(samples, T, 0))
    cache = create_vecchia_cache_gpu(I, J, samples, T(lambda), format, uplo)

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
        # The bounds may be given on the CPU or on the GPU.
        w_lvar .= log.(copyto!(similar(samples, T, cache.n), lvar_diag))
    else
        w_lvar .= log(T(1e-10))
    end

    if !isnothing(uvar_diag)
        w_uvar .= log.(copyto!(similar(samples, T, cache.n), uvar_diag))
    else
        w_uvar .= log(T(1e10))
    end

    view(x0, cache.diagL) .= 1.0

    meta = NLPModelMeta{T, S}(
        nvar,
        ncon = ncon,
        x0 = x0,
        name = name,
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

function create_vecchia_cache_gpu(I::Vector{Int}, J::Vector{Int}, samples::AbstractMatrix{T},
                                  lambda::T, format::Symbol, uplo::Symbol) where {T}
    S = typeof(similar(samples, T, 0))
    to_device(v) = copyto!(similar(samples, Int, length(v)), v)
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

    offsets = cumsum([0; m[1:end-1]]) |> to_device
    B = [similar(samples, T, 0, 0)]

    rowsL = to_device(rowsL)
    colptrL = to_device(colptrL)
    m = to_device(m)

    hess_obj_vals = S(undef, nnzh_tri_obj)
    vecchia_build_B!(B, samples, lambda, rowsL, colptrL, hess_obj_vals, n, m)

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
        m, offsets, B, nnzh_tri_obj,
        nnzh_tri_lag, hess_obj_vals,
        buffer,
    )
end
