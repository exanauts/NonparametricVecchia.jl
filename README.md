# NonparametricVecchia

[![docs-dev][docs-dev-img]][docs-dev-url] [![ci][ci-img]][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/NonparametricVecchia.jl/dev
[ci-img]: https://github.com/exanauts/NonparametricVecchia.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/exanauts/NonparametricVecchia.jl/actions/workflows/CI.yml

## Overview

**NonparametricVecchia.jl** is a Julia package that approximates the inverse
cholesky factor of a Gaussian process covariance matrix via a nonparametric
optimization process.  The nonzero entries of the inverse cholesky, `L`, are
determined via the Vecchia Approximation.  The values of `L` are recovered via
optimizing the joint probability distribution function (mean zero) to best match
the given samples.

## Installation

This package is not registered but can be installed and tested through the Julia
package manager:

```julia
julia> ]
pkg> add https://github.com/exanauts/NonparametricVecchia.jl.git
pkg> test NonparametricVecchia
```

## Usage

A `VecchiaModel` is an [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
model, so it can be solved with any compatible solver (MadNLP, Ipopt, Uno, ...).
The sparsity pattern of the factor is given as a `LowerTriangular` or
`UpperTriangular` sparse matrix, and the samples are stored row-wise
(one replicate per row):

```julia
using NonparametricVecchia, MadNLP, SparseArrays, LinearAlgebra

n, k, m = 1000, 10, 200                       # size, conditioning points, replicates
samples = randn(m, n)
pattern = LowerTriangular(spdiagm([-j => trues(n - j) for j in 0:k]...))

nlp = VecchiaModel(pattern, samples)
result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
L = recover_factor(nlp, result.solution)       # Σ⁻¹ ≈ L * L'
```

### Structure-exploiting KKT system

With `kkt_system=VecchiaKKTSystem`, MadNLP does not use a sparse linear solver:
the Newton systems of the interior-point method are solved by block elimination,
with dense Cholesky factorizations of the diagonal blocks of the Hessian (one block
per column of the factor, factorized once and reused across iterations) and
independent `2 × 2` systems. This is usually much faster than the default
`SparseKKTSystem`. The `linear_solver` option of MadNLP is ignored.

### GPU support

NVIDIA and AMD GPUs are supported through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (v6)
and [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl).
When the samples are stored in a `CuMatrix` or a `ROCMatrix`, the model is built on the GPU,
and `VecchiaKKTSystem` performs the factorizations and the solves with GPU kernels:

```julia
using CUDA, MadNLPGPU                          # or: using AMDGPU, MadNLPGPU

nlp = VecchiaModel(pattern, CuMatrix(samples)) # or: ROCMatrix(samples)
result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
L = recover_factor(nlp, result.solution)       # CuSparseMatrixCSC or ROCSparseMatrixCSC
```

If the number of replicates `m` is smaller than the number of nonzeros per column
(`k + 1`), the blocks of the Hessian are singular. Adding a ridge penalty with the
keyword `lambda` (for instance `VecchiaModel(pattern, samples; lambda=1e-6)`) makes
them positive definite, so that they are factorized only once.

See the [documentation](https://exanauts.github.io/NonparametricVecchia.jl/dev) for more details.
