# Solving with MadNLP and `VecchiaKKTSystem`

The optimization problem behind a [`VecchiaModel`](@ref) has a very particular
structure: the Hessian of the objective is block diagonal, with one dense block
`Hⱼ` per column of the factor, and each constraint only couples the diagonal entry
of a column with its logarithm `wⱼ`.
`VecchiaKKTSystem` is a custom KKT system for [MadNLP.jl](https://github.com/madsuite-org/MadNLP.jl)
that exploits this structure to solve the Newton systems of the interior-point method:

1. the blocks `Hⱼ` are factorized with a dense Cholesky factorization;
2. the remaining system in the variables `w` and the multipliers of the constraints
   reduces to `n` independent `2 × 2` systems;
3. the inertia of the KKT matrix, needed by the interior-point method, is computed
   analytically from these `2 × 2` systems.

No sparse linear solver is involved, and the `linear_solver` option of MadNLP is ignored.
Since the blocks `Hⱼ` do not depend on the iterate, they are factorized once and reused
across the iterations, unless MadNLP has to regularize them.

## Usage on CPU

```@example kkt
using NonparametricVecchia
using MadNLP
using LinearAlgebra
using SparseArrays

n, k, m = 500, 10, 200
samples = randn(m, n)
pattern = LowerTriangular(spdiagm([-j => trues(n - j) for j in 0:k]...))
nlp = VecchiaModel(pattern, samples)

result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
T = recover_factor(nlp, result.solution)
nothing # hide
```

The solution is the same as the one computed with the default KKT system of MadNLP
and a sparse linear solver, but the Newton steps are much cheaper:

```@example kkt
result_default = madnlp(nlp; print_level=MadNLP.ERROR)
result_vecchia = madnlp(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
norm(result_default.solution - result_vecchia.solution, Inf)
```

## Usage on GPU

`VecchiaKKTSystem` also runs on NVIDIA GPUs with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (v6)
and on AMD GPUs with [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl).
The factorizations and the solves with the blocks `Hⱼ` are performed by GPU kernels written with
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl), one block per thread,
and the model itself (objective, gradient, Hessian) is evaluated on the GPU.
It suffices to store the samples in a `CuMatrix` or a `ROCMatrix`, and to load
[MadNLPGPU.jl](https://github.com/madsuite-org/MadNLP.jl/tree/master/lib/MadNLPGPU):

```julia
using CUDA, MadNLPGPU

nlp = VecchiaModel(pattern, CuMatrix(samples))
result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
T = recover_factor(nlp, result.solution)  # CuSparseMatrixCSC
```

```julia
using AMDGPU, MadNLPGPU

nlp = VecchiaModel(pattern, ROCMatrix(samples))
result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
T = recover_factor(nlp, result.solution)  # ROCSparseMatrixCSC
```

## Bounds and regularization

The keyword arguments `lvar_diag` and `uvar_diag` of [`VecchiaModel`](@ref) bound the
diagonal entries of the factor. They are imposed on the variables `w = log(diag(T))`,
so that the blocks `Hⱼ` remain independent of the iterate.

If the number of replicates `m` is smaller than the number of nonzeros in a column of
the factor, the corresponding block `Hⱼ` is singular. MadNLP then regularizes the
KKT system at every iteration, and the blocks have to be refactorized each time.
A small ridge penalty, set with the keyword `lambda`, makes all blocks positive definite:

```@example kkt
samples_small = randn(5, n)  # only 5 replicates for 11 nonzeros per column
nlp_small = VecchiaModel(pattern, samples_small; lambda=1e-2)
result_small = madnlp(nlp_small; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
result_small.status
```

## Limitations

- `VecchiaKKTSystem` only supports a `VecchiaModel` solved with the exact Hessian
  (no quasi-Newton approximation).
- The bounds on the diagonal entries must satisfy `lvar_diag < uvar_diag`
  (fixed variables are not supported).
