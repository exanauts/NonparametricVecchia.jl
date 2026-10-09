# NonparametricVecchia

[![docs-dev][docs-dev-img]][docs-dev-url] [![ci][ci-img]][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://exanauts.github.io/NonparametricVecchia.jl/dev
[ci-img]: https://github.com/exanauts/NonparametricVecchia.jl/actions/workflows/CI.yml/badge.svg
[ci-url]: https://github.com/exanauts/NonparametricVecchia.jl/actions/workflows/CI.yml

## Overview

**NonparametricVecchia.jl** is a Julia package that estimates a sparse inverse
Cholesky factor of the covariance matrix of a Gaussian process, without assuming a
parametric covariance function. It computes a sparse triangular factor `T` such that
`T * T'` approximates the inverse of the covariance matrix, by maximizing the likelihood
of the given samples (mean zero). The sparsity pattern of `T` is chosen a priori,
typically from Vecchia's approximation.

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
T = recover_factor(nlp, result.solution)       # Σ⁻¹ ≈ T * T'
```

### Structure-exploiting KKT system

With `kkt_system=VecchiaKKTSystem`, MadNLP does not use a sparse linear solver:
the Newton systems of the interior-point method are solved by block elimination,
with dense Cholesky factorizations of the diagonal blocks of the Hessian (one block
per column of the factor, factorized once and reused across iterations) and
independent `2 × 2` systems. This is usually much faster than the default
`SparseKKTSystem`. The `linear_solver` option of MadNLP is ignored.

If the number of replicates `m` is smaller than the number of nonzeros per column
(`k + 1`), the blocks of the Hessian are singular. Adding a ridge penalty with the
keyword `lambda` (for instance `VecchiaModel(pattern, samples; lambda=1e-6)`) makes
them positive definite, so that they are factorized only once.

### GPU support

NVIDIA and AMD GPUs are supported through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
and [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl).
When the samples are stored in a `CuMatrix` or a `ROCMatrix`, the model is built on the GPU,
and `VecchiaKKTSystem` performs the factorizations and the solves with GPU kernels:

```julia
using CUDA, MadNLPGPU                          # or: using AMDGPU, MadNLPGPU

nlp = VecchiaModel(pattern, CuMatrix(samples)) # or: ROCMatrix(samples)
result = madnlp(nlp; kkt_system=VecchiaKKTSystem)
T = recover_factor(nlp, result.solution)       # CuSparseMatrixCSC or ROCSparseMatrixCSC
```

See the [documentation](https://exanauts.github.io/NonparametricVecchia.jl/dev) for more details.
