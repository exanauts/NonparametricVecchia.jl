# `GPUMatrix` is the matrix type of the GPU backend under test (CuMatrix or ROCMatrix).
function gpu_factor_to_cpu(W)
    n = size(W, 1)
    return SparseMatrixCSC(n, n, Array(W.colPtr), Array(W.rowVal), Array(W.nzVal))
end

@testset "CPU_Compatible_GPU -- $GPUMatrix" begin
    samples = gensamples(100, 75)

    @testset "uplo = $uplo, lambda = $lambda" for (uplo, pattern) in ((:U, banded_U(100, 5)), (:L, banded_L(100, 5))),
                                                  lambda in (0.0, 1e-8, 1.0)
        # Reference on CPU with the default KKT system
        model  = VecchiaModel(pattern, samples; lambda)
        output = madnlp(model; print_level=MadNLP.ERROR)
        W_cpu  = recover_factor(model, output.solution)

        # GPU with VecchiaKKTSystem
        model  = VecchiaModel(pattern, GPUMatrix(samples); lambda)
        output = madnlp(model; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        @test output.status == MadNLP.SOLVE_SUCCEEDED
        W_gpu  = recover_factor(model, output.solution)

        @test norm(W_cpu - gpu_factor_to_cpu(W_gpu)) ≤ 1e-4
    end
end
