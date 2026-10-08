```@example VecchiaModel_L
using NonparametricVecchia
using LinearAlgebra
using SparseArrays
using NLPModelsIpopt

n = 40
number_of_samples = 100
samples = randn(number_of_samples, n)

P = ones(n, n)
P = tril(P)
P = sparse(P)
I, J, V = findnz(P)
nlp_L = VecchiaModel(I, J, samples; format=:coo, uplo=:L)
output = ipopt(nlp_L)
L = recover_factor(nlp_L, output.solution)
```

```@example VecchiaModel_U
using NonparametricVecchia
using LinearAlgebra
using SparseArrays
using NLPModelsIpopt

n = 40
number_of_samples = 100
samples = randn(number_of_samples, n)

P = ones(n, n)
P = triu(P)
P = sparse(P)
I, J, V = findnz(P)
nlp_U = VecchiaModel(I, J, samples; format=:coo, uplo=:U)
output = ipopt(nlp_U)
U = recover_factor(nlp_U, output.solution)
```

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
