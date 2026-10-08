module NonparametricVecchia

using KernelAbstractions
using LinearAlgebra
using NLPModels
using SparseArrays

include("VecchiaModel.jl")
include("api.jl")

"""
    VecchiaKKTSystem

Structure-exploiting KKT system for solving a `VecchiaModel` with MadNLP.

The Newton system of the interior-point method is solved by block elimination:
the dense diagonal blocks `Hⱼ` of the Hessian of the objective are factorized with a
Cholesky factorization, and the remaining system in `(Δw, Δλ)` is made of `n`
independent `2 × 2` systems. No sparse linear solver is required, and the
`linear_solver` option of MadNLP is ignored.

The implementation is available once MadNLP.jl is loaded:
```julia
using NonparametricVecchia, MadNLP
nlp = VecchiaModel(L_pattern, samples)
madnlp(nlp; kkt_system=VecchiaKKTSystem)
```
"""
struct VecchiaKKTSystem end

export VecchiaModel, VecchiaKKTSystem, recover_factor

end
