module NonparametricVecchiaAMDGPUExt

using NonparametricVecchia
using AMDGPU

function NonparametricVecchia.VecchiaModel(I::Vector{Int}, J::Vector{Int}, samples::ROCMatrix{T}; kwargs...) where T
    return NonparametricVecchia.vecchia_model_gpu(I, J, samples, "nonparametric_vecchia_amdgpu"; kwargs...)
end

function NonparametricVecchia.recover_factor(nlp::VecchiaModel{T,<:ROCVector{T}}, solution::ROCVector{T}) where T
    n = nlp.cache.n
    colptr = nlp.cache.colptrL
    rowval = nlp.cache.rowsL
    nnz_factor = length(rowval)
    nzval = solution[1:nnz_factor]
    factor = AMDGPU.rocSPARSE.ROCSparseMatrixCSC(colptr, rowval, nzval, (n, n))
    return factor
end

end  # end module
