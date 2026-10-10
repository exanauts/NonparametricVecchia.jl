# Measuring the computational cost

This tutorial explains how to measure the time spent in each phase of the estimation, and how
to check the complexity of the method.
With `n` columns, `k` conditioning points per column and `m` replicates, the cost of the
estimation splits into the following phases:

| Phase | Where | Cost | How to measure it |
|:------|:------|:-----|:------------------|
| Assembly of the blocks `Hⱼ` of the Hessian | `VecchiaModel` constructor (`VecchiaCache`) | `O(n m k²)`, once | `@elapsed VecchiaModel(...)` |
| Initialization of MadNLP (including the creation of the KKT system) | `MadNLPSolver` constructor | `O(n k²)`, once | `solver.cnt.init_time` |
| Evaluations of the objective, gradient and Hessian | `VecchiaModel` (NLPModels API) | `O(n k²)` per evaluation | `solver.cnt.eval_function_time` |
| Cholesky factorizations of the blocks `Hⱼ` | `VecchiaKKTSystem` | `O(n k³)`, about twice | `factorization_time` |
| `2 × 2` systems and inertia | `VecchiaKKTSystem` | `O(n)` per iteration | `schur_time` |
| Backsolves | `VecchiaKKTSystem` | `O(n k²)` per backsolve | `backsolve_time` |
| Products with the KKT matrix (iterative refinement) | `VecchiaKKTSystem` | `O(n k²)` per product | `product_time` |

The blocks `Hⱼ` do not depend on the iterate, so they are assembled once, when the model is built,
and the objective and its derivatives are then evaluated from them without going back to the samples.
Unlike a sparse linear solver, [`VecchiaKKTSystem`](@ref) has no symbolic analysis phase:
the block structure of the KKT systems is known in advance, and only the offsets of the blocks
are computed when the KKT system is created.
The blocks are factorized when MadNLP initializes the multipliers (with a zero Hessian) and once
with the actual Hessian; they are then reused at every iteration, unless MadNLP has to regularize
the KKT system. All timings and counters of the linear algebra are returned by [`vecchia_kkt_stats`](@ref).

## Breakdown of a solve

We consider `n` points on a line with an exponential covariance function, whose samples are
generated with an autoregressive recursion, and a banded sparsity pattern with `k` conditioning
points per column.

```@example timings
using NonparametricVecchia
using MadNLP
using LinearAlgebra
using SparseArrays
using Random

# Samples of a process with covariance exp(-|i - j| / ℓ) on a regular grid, one replicate per row
function ar1_samples(n, m; ℓ = 10.0, rng = Xoshiro(0))
    ρ = exp(-1 / ℓ)
    Y = Matrix{Float64}(undef, m, n)
    Y[:, 1] .= randn(rng, m)
    for j in 2:n
        Y[:, j] .= ρ .* view(Y, :, j-1) .+ sqrt(1 - ρ^2) .* randn(rng, m)
    end
    return Y
end

banded_pattern(n, k) = LowerTriangular(spdiagm([-j => trues(n - j) for j in 0:k]...))

# Run once on a small problem, so that the timings below do not include compilation
let nlp = VecchiaModel(banded_pattern(100, 2), ar1_samples(100, 10))
    MadNLP.solve!(MadNLPSolver(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR))
end
nothing # hide
```

The assembly of the blocks `Hⱼ` is measured when the model is built, and the other phases are
measured during the solve. The initialization of MadNLP includes the creation of the KKT system,
and the rest of MadNLP corresponds to the operations of the interior-point method itself
(line search, update of the iterates and of the barrier parameter, convergence tests):

```@example timings
n, k, m = 20_000, 10, 200
samples = ar1_samples(n, m)
pattern = banded_pattern(n, k)

time_assembly = @elapsed (nlp = VecchiaModel(pattern, samples))

solver = MadNLPSolver(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
MadNLP.solve!(solver)
stats = vecchia_kkt_stats(solver)

time_linear_algebra = stats.factorization_time + stats.schur_time + stats.backsolve_time + stats.product_time
time_madnlp = solver.cnt.total_time - solver.cnt.init_time - solver.cnt.eval_function_time - time_linear_algebra
phases = [
    "Assembly of the blocks Hⱼ"     => time_assembly,
    "Initialization of MadNLP"      => solver.cnt.init_time,
    "Function evaluations"          => solver.cnt.eval_function_time,
    "Factorizations of the blocks"  => stats.factorization_time,
    "2 × 2 systems and inertia"     => stats.schur_time,
    "Backsolves"                    => stats.backsolve_time,
    "Products with the KKT matrix"  => stats.product_time,
    "Rest of MadNLP"                => time_madnlp,
]
for (phase, time) in phases
    println(rpad(phase, 32), round(1000 * time, digits=2), " ms")
end
```

The counters show how often each operation is performed during the `solver.cnt.k` iterations,
and that the blocks are only factorized about twice, even if MadNLP requests a factorization at
every iteration:

```@example timings
(iterations = solver.cnt.k, nfactorizations = stats.nfactorizations, nbacksolves = stats.nbacksolves,
 nproducts = stats.nproducts, nblocks = stats.nblocks, nblocks_factorized = stats.nblocks_factorized)
```

## Checking the complexity

To check the complexity of each phase, we vary one parameter at a time and estimate the exponent
of the corresponding growth with a linear fit in log-log scale.
Since the factorization of a block `Hⱼ` of size `k+1` costs `O((k+1)³)`, the parameter used for the
fits in `k` is `k+1`. The script below is not run when the documentation is built, because the
measurements require larger problems and a quiet machine.

```julia
# Exponent α of t ≈ C xᵅ, by a least-squares fit in log-log scale
function exponent(x, t)
    lx, lt = log.(x), log.(t)
    lx_mean, lt_mean = sum(lx) / length(lx), sum(lt) / length(lt)
    return sum((lx .- lx_mean) .* (lt .- lt_mean)) / sum((lx .- lx_mean) .^ 2)
end

function measure(n, k, m; repetitions = 3)
    samples = ar1_samples(n, m)
    pattern = banded_pattern(n, k)
    assembly = minimum(@elapsed(VecchiaModel(pattern, samples)) for _ in 1:repetitions)
    nlp = VecchiaModel(pattern, samples)
    solver = MadNLPSolver(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
    MadNLP.solve!(solver)
    stats = vecchia_kkt_stats(solver)
    return (assembly = assembly,
            factorization = stats.factorization_time,
            backsolve = stats.backsolve_time / stats.nbacksolves)
end

# Exponents in n (expected: 1 for all phases)
ns = [25_000, 50_000, 100_000, 200_000]
results = [measure(n, 10, 200) for n in ns]
exponent(ns, [r.assembly for r in results]), exponent(ns, [r.factorization for r in results]),
exponent(ns, [r.backsolve for r in results])

# Exponents in k+1 (expected: 2 for the assembly, 3 for the factorization, 2 for the backsolves)
ks = [10, 20, 40, 80]
results = [measure(20_000, k, 200) for k in ks]
exponent(ks .+ 1, [r.assembly for r in results]), exponent(ks .+ 1, [r.factorization for r in results]),
exponent(ks .+ 1, [r.backsolve for r in results])

# Exponent in m (expected: 1 for the assembly, 0 for the factorization and the backsolves)
ms = [1_000, 2_000, 4_000, 8_000]
results = [measure(20_000, 10, m) for m in ms]
exponent(ms, [r.assembly for r in results])
```

The complexity of the paper describes the asymptotic behavior, and the measured exponents can be
smaller for small problems:

- the factorization of the blocks also copies and compares the entries of `Hⱼ` to detect whether
  a refactorization is needed, which costs `O(n k²)`, so its exponent in `k+1` only approaches 3
  when `k` is large enough;
- the backsolves and the products also perform `O(n)` operations on vectors, which dominate for
  small `k`;
- the time measured for the assembly also includes the conversion of the sparsity pattern and the
  allocation of the model, which do not depend on `m`, so its exponent in `m` approaches 1 when
  `m k²` is large;
- when the blocks no longer fit in the cache of the processor, the memory bandwidth can make
  the exponents larger than expected.

Fitting the exponents on the largest values of each parameter gives a better estimate of the
asymptotic behavior. The same measurements can be done on GPU, since [`vecchia_kkt_stats`](@ref)
is also available for models built from a `CuMatrix` or a `ROCMatrix`. The factorizations and the
solves synchronize the GPU, but the construction of the model must be followed by
`CUDA.synchronize()` or `AMDGPU.synchronize()` before measuring its time.
