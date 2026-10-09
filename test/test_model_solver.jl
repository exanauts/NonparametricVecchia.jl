@testset "different_model_solvers" begin
    samples = gensamples(100, 75)

    @testset "uplo = $uplo, lambda = $lambda" for (uplo, pattern) in ((:U, banded_U(100, 5)), (:L, banded_L(100, 5))),
                                                  lambda in (0.0, 1.0)
        nlp = VecchiaModel(pattern, samples; lambda)

        # Reference: MadNLP with its default KKT system
        ref = madnlp(nlp; print_level=MadNLP.ERROR)
        @test ref.status == MadNLP.SOLVE_SUCCEEDED

        @testset "Ipopt" begin
            stats = ipopt(nlp; print_level=0)
            @test stats.status == :first_order
            @test norm(stats.solution - ref.solution, Inf) ≤ 1e-6
            @test stats.objective ≈ ref.objective rtol=1e-8
        end

        @testset "Uno" begin
            stats = uno(nlp)
            @test stats.status == :first_order
            @test norm(stats.solution - ref.solution, Inf) ≤ 1e-6
            @test stats.objective ≈ ref.objective rtol=1e-8
        end
    end
end
