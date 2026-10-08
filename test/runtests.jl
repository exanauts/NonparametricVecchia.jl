using NonparametricVecchia
using Test
using JuMP
using Ipopt
using NLPModelsJuMP
using NLPModelsIpopt
using UnoSolver
using MadNLP
using MadNLPGPU
using SparseArrays
using LinearAlgebra
using CUDA
using AMDGPU
using NLPModels
using NLPModelsTest
using StableRNGs
using Random

import NonparametricVecchia: VecchiaCache

function gensamples(n, nsample)
  K = [exp(-abs(j-k)/10)*(1+abs(j-k)/10) for j in 1:n, k in 1:n]
  samps_cols = cholesky(Symmetric(K)).L*randn(StableRNG(1234), n, nsample)
  permutedims(samps_cols)
end

function banded_U(n, k)
  sp = spzeros(Bool, n, n)
  for offset in 0:k
    for j in (1+offset):n
      sp[j-offset,j] = true
    end
  end
  UpperTriangular(sp)
end
banded_L(n, k) = LowerTriangular(tril(banded_U(n, k).data'))

include("Jump_models.jl")
include("models.jl")

include("test_jump.jl")
include("test_memory_allocation_cpu.jl")
# include("test_cpu_compatible_with_jump.jl")

if CUDA.functional()
    # include("test_gpu_compatible_with_jump.jl")
    GPUMatrix = CuMatrix
    include("test_cpu_compatible_with_gpu.jl")
    include("test_memory_allocation_gpu.jl")
end

if AMDGPU.functional()
    GPUMatrix = ROCMatrix
    include("test_cpu_compatible_with_gpu.jl")
end

include("test_model_solver.jl")
include("test_kkt_system.jl")
include("test_coo_vs_csc.jl")

using Vecchia
using StaticArrays

include("vecchia.jl")

using HSL
using MadNLPHSL

if LIBHSL_isfunctional()
    include("test_linear_solver.jl")
end
