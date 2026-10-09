# [NonparametricVecchia.jl documentation](@id Home)

## Overview

This package estimates the covariance structure of a Gaussian random field from i.i.d. replicates, without assuming a parametric covariance function.
It computes a sparse triangular factor `T` such that `T * T'` approximates the inverse of the covariance matrix, by maximum likelihood estimation.
The sparsity pattern of `T` is chosen a priori, typically from Vecchia's approximation, where each column only involves a small set of conditioning points.
The estimation problem is solved with interior-point methods, and its structure makes it highly efficient for large-scale problems, on CPUs and GPUs.

## Features

- `VecchiaModel`: the maximum likelihood estimation problem as an [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) model, which can be solved with MadNLP, Ipopt, Uno, ...
- `VecchiaKKTSystem`: a structure-exploiting KKT system for [MadNLP.jl](https://github.com/madsuite-org/MadNLP.jl), which solves the Newton systems with dense block Cholesky factorizations instead of a sparse linear solver.
- GPU support for NVIDIA and AMD GPUs through [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) (v6) and [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl): the model and `VecchiaKKTSystem` run entirely on the GPU.
- Integration with [Vecchia.jl](https://github.com/cgeoga/Vecchia.jl) to choose the ordering and the conditioning sets.

## Installation

```julia
julia> ]
pkg> add https://github.com/exanauts/NonparametricVecchia.jl.git
pkg> test NonparametricVecchia
```
