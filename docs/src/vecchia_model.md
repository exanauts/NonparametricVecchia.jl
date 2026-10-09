# Building and solving a `VecchiaModel`

A [`VecchiaModel`](@ref) is the maximum likelihood estimation problem whose solution is a sparse
triangular factor `L` such that `L * L'` approximates the inverse of the covariance matrix of the samples.
It is built from two ingredients:

- the sparsity pattern of the factor, given as a `LowerTriangular` or `UpperTriangular` sparse matrix,
  or as row and column indices;
- a matrix of samples of size `m × n`, with one replicate per row.

The model is an `AbstractNLPModel` of [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl),
so it can be solved with any compatible solver, such as MadNLP, Ipopt or Uno.
The factor is then recovered from the solution with [`recover_factor`](@ref).

## Sparsity pattern and samples

As a first example, we consider `n` points on a line with an exponential covariance
function, and a banded sparsity pattern: each column of the factor involves the `k`
previous points. The samples are stored in a matrix of size `m × n`, with one replicate per row.

```@example basics
using NonparametricVecchia
using LinearAlgebra
using SparseArrays

n, k, m = 200, 5, 100
K = [exp(-abs(i - j) / 10) for i in 1:n, j in 1:n]  # covariance matrix
samples = Matrix((cholesky(K).L * randn(n, m))')     # one replicate per row

pattern = LowerTriangular(spdiagm([-j => trues(n - j) for j in 0:k]...))
nlp = VecchiaModel(pattern, samples)
nothing # hide
```

The same model can be built from the row and column indices of the nonzeros (COO format),
and an upper triangular factor is obtained with an `UpperTriangular` pattern or with `uplo=:U`.
For an upper triangular factor, the diagonal entry of each column is its last nonzero instead of its first one.

```@example basics
rows, cols, _ = findnz(pattern.data)
nlp_coo = VecchiaModel(rows, cols, samples; format=:coo, uplo=:L)

pattern_U = UpperTriangular(sparse(pattern.data'))
nlp_U = VecchiaModel(pattern_U, samples)
nothing # hide
```

The model is solved with MadNLP and [`VecchiaKKTSystem`](@ref), and the factor is recovered with [`recover_factor`](@ref):

```@example basics
using MadNLP

result = madnlp(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
L = recover_factor(nlp, result.solution)
result.status
```

Since `L * L'` approximates the inverse of `K`, the product `L' * K * L` approximates the identity.
The relative error decreases as the number of replicates increases:

```@example basics
for m in (100, 1_000, 10_000)
    samples_m = Matrix((cholesky(K).L * randn(n, m))')
    nlp_m = VecchiaModel(pattern, samples_m)
    result_m = madnlp(nlp_m; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
    L_m = recover_factor(nlp_m, result_m.solution)
    println("m = ", m, ": ", norm(L_m' * K * L_m - I) / sqrt(n))
end
```

Any other solver of the JuliaSmoothOptimizers ecosystem can be used as well, for instance
Ipopt with `using NLPModelsIpopt; ipopt(nlp)`, or Uno with `using UnoSolver; uno(nlp)`.

## Sparsity pattern from Vecchia.jl

In practice, the sparsity pattern of the factor comes from the Vecchia approximation:
the nonzeros of each column correspond to a small set of conditioning points.
With [Vecchia.jl](https://github.com/cgeoga/Vecchia.jl), a `VecchiaModel` can be built
directly from the locations of the observations, an ordering, and a design of the
conditioning sets. Following the conventions of Vecchia.jl, the data are then given with one
replicate per column. The model is solved with MadNLP and [`VecchiaKKTSystem`](@ref).

```@example Vecchia
using NonparametricVecchia
using Vecchia
using StaticArrays
using LinearAlgebra

# Generate some fake data with an exponential covariance function.
# Note that each _column_ of sim is an iid replicate, in keeping with formatting
# standards of the many R packages for GPs. In this demonstration, n is small
# and the number of replicates is large to demonstrate asymptotic correctness.
pts = rand(SVector{2,Float64}, 100)
z   = randn(length(pts), 1000)
K   = Symmetric([exp(-norm(x-y)) for x in pts, y in pts])
sim = cholesky(K).L * z

# Create a VecchiaModel using the options specified by Vecchia.jl, in
# particular choosing an ordering (in this case RandomOrdering()) and a
# conditioning set design (in this case KNNConditioning(10)). This also returns
# the permutation for you to use with subsequent data operations. See the docs
# and myriad extensions of Vecchia.jl for more information on ordering and
# conditioning set design options.
(perm, nlp) = VecchiaModel(pts, sim, RandomOrdering(), KNNConditioning(10);
                           lvar_diag=fill(inv(sqrt(1.0)),  length(pts)),
                           uvar_diag=fill(inv(sqrt(1e-2)), length(pts)),
                           lambda=1e-3)

# Now bring in some optimizers and fit the nonparametric model. This gives a U
# such that Σ^{-1} ≈ U*U', where Σ is the covariance matrix for each column of sim.
# The option `kkt_system=VecchiaKKTSystem` exploits the block structure of the KKT
# systems: the Newton steps are computed with dense Cholesky factorizations of the
# blocks of the Hessian, without any sparse linear solver.
using MadNLP
result = madnlp(nlp; tol=1e-10, kkt_system=VecchiaKKTSystem)
U      = UpperTriangular(recover_factor(nlp, result.solution))

# KL divergence from the true covariance:
K_perm = K[perm, perm]
kl     = (tr(U'*K_perm*U) - length(pts)) + (-2*logdet(U) - logdet(K_perm))

# compare that with what you get from a parametric Vecchia model using the
# correct kernel and the same permutation.
para_vecchia = VecchiaApproximation(pts[perm], (x,y,p)->exp(-norm(x-y)), sim[perm,:];
                                    ordering=NoPermutation())
para_U  = rchol(para_vecchia, Float64[]).U
para_kl = (tr(para_U'*K_perm*para_U) - length(pts)) + (-2*logdet(para_U) - logdet(K_perm))

println("KL divergence with true parametric kernel:  ", para_kl)
println("KL divergence with nonparametric estimator: ", kl)
```
