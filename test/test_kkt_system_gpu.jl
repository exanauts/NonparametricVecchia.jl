@testset "VecchiaKKTSystem -- GPU" begin
    samples = gensamples(100, 75)

    @testset "uplo = $uplo, lambda = $lambda" for (uplo, pattern) in ((:U, banded_U(100, 5)), (:L, banded_L(100, 5))),
                                                  lambda in (0.0, 1e-8, 1.0)
        # Reference on CPU with the default KKT system
        model  = VecchiaModel(pattern, samples; lambda)
        output = madnlp(model; print_level=MadNLP.ERROR)
        W_cpu  = recover_factor(model, output.solution)

        # GPU with VecchiaKKTSystem
        model  = VecchiaModel(pattern, CuMatrix(samples); lambda)
        output = madnlp(model; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        @test output.status == MadNLP.SOLVE_SUCCEEDED
        W_gpu  = recover_factor(model, output.solution)

        @test norm(W_cpu - SparseMatrixCSC(W_gpu)) ≤ 1e-4
    end
end
