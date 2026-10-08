module NonparametricVecchiaCUDAExt

using NonparametricVecchia
using CUDA
using CUDA.cuSPARSE: CuSparseMatrixCSC

function NonparametricVecchia.VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::CuMatrix{T}; kwargs...) where T
    return NonparametricVecchia.vecchia_model_gpu(I, J, samples, "nonparametric_vecchia_cuda"; kwargs...)
end

function NonparametricVecchia.recover_factor(nlp::VecchiaModel{T,<:CuVector{T}}, solution::CuVector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = CuSparseMatrixCSC(colptr, rowval, nzval, (n, n))
    return factor
end

end  # end module
