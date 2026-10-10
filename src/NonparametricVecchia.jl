module NonparametricVecchia

using LinearAlgebra
using NLPModels
using SparseArrays

include("VecchiaModel.jl")
include("api.jl")
include("kkt_blocks.jl")
include("gpu.jl")

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

"""
    stats = vecchia_kkt_stats(solver::MadNLPSolver)

Return the cumulative timings (in seconds) and counters of the linear algebra of
a `MadNLPSolver` that uses `kkt_system=VecchiaKKTSystem`, as a `NamedTuple` with the fields:

- `factorization_time`: Cholesky factorizations of the blocks `Hⱼ` (and computation of the columns `Hⱼ⁻¹ e_{dⱼ}`);
- `schur_time`: assembly of the `2 × 2` systems and computation of the inertia;
- `backsolve_time`: solves of the KKT systems (one block solve and `O(n)` operations each);
- `product_time`: products with the KKT matrix, used by the iterative refinement of MadNLP;
- `nfactorizations`, `nbacksolves`, `nproducts`: number of calls of each operation;
- `nblocks`: number of blocks `Hⱼ` (columns of the factor);
- `nblocks_factorized`: total number of block factorizations. A block is only refactorized
  when its entries or its regularization changed, so this number is usually close to
  `2 * nblocks`, even if `nfactorizations` is much larger.
"""
function vecchia_kkt_stats end

export VecchiaModel, VecchiaKKTSystem, recover_factor, vecchia_kkt_stats

end
