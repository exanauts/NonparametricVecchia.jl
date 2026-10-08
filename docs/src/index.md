# [NonparametricVecchia.jl documentation](@id Home)

## Overview

This package computes an approximate Cholesky factorization of a covariance matrix using a Maximum Likelihood Estimation (MLE) approach.
The Cholesky factor is computed via the Vecchia approximation, which is sparse and approximately banded, making it highly efficient for large-scale problems.

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
