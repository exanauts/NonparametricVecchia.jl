@testset "VecchiaKKTSystem" begin
    samples = gensamples(100, 75)

    @testset "Linear algebra -- uplo = $uplo" for (uplo, pattern) in ((:U, banded_U(100, 5)), (:L, banded_L(100, 5)))
        nlp = VecchiaModel(pattern, samples; lambda=1e-2)
        solver = MadNLPSolver(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        kkt = solver.kkt
        nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
        nlb, nub = length(kkt.ind_lb), length(kkt.ind_ub)
        rng = StableRNG(2025)

        hrows, hcols = hess_structure(nlp)
        jrows, jcols = jac_structure(nlp)

        for trial in 1:3
            # Random iterate, with possibly negative multipliers
            x = randn(rng, nvar)
            y = randn(rng, ncon)
            hess_coord!(nlp, x, y, kkt.hess)
            jac_coord!(nlp, x, kkt.jac)
            kkt.reg .= rand(rng, nvar)
            kkt.du_diag .= trial == 1 ? 0.0 : -rand(rng, ncon)
            kkt.l_lower .= rand(rng, nlb)
            kkt.u_lower .= rand(rng, nub)
            kkt.l_diag .= .-(1.0 .+ rand(rng, nlb))
            kkt.u_diag .= .-(1.0 .+ rand(rng, nub))
            MadNLP._set_aug_diagonal!(kkt)
            MadNLP.factorize!(kkt.linear_solver)

            # Reduced KKT matrix
            H = sparse(hrows, hcols, kkt.hess, nvar, nvar)
            H = H + H' - Diagonal(diag(H))
            J = sparse(jrows, jcols, kkt.jac, ncon, nvar)
            K = Matrix([H+Diagonal(kkt.pr_diag) J'; J Diagonal(kkt.du_diag)])

            # Inertia
            λ = eigvals(Symmetric(K))
            @test MadNLP.inertia(kkt.linear_solver) == (count(>(0), λ), 0, count(<(0), λ))

            # Product with the unreduced KKT matrix
            v = MadNLP.UnreducedKKTVector(kkt)
            randn!(rng, MadNLP.full(v))
            Kv = MadNLP.UnreducedKKTVector(kkt)
            mul!(Kv, kkt, v)
            Kreg = Matrix([H+Diagonal(kkt.reg) J'; J Diagonal(kkt.du_diag)])
            expected = Kreg * MadNLP.primal_dual(v)
            view(expected, kkt.ind_lb) .-= MadNLP.dual_lb(v)
            view(expected, kkt.ind_ub) .+= MadNLP.dual_ub(v)
            @test MadNLP.primal_dual(Kv) ≈ expected

            # Solution of the unreduced KKT system
            b = MadNLP.UnreducedKKTVector(kkt)
            randn!(rng, MadNLP.full(b))
            d = copy(b)
            MadNLP.solve_kkt!(kkt, d)
            r = copy(b)
            mul!(r, kkt, d, -1.0, 1.0)
            @test norm(MadNLP.full(r)) ≤ 1e-8 * norm(MadNLP.full(b))
        end
    end

    @testset "Solve -- uplo = $uplo, lambda = $lambda" for (uplo, pattern) in ((:U, banded_U(100, 5)), (:L, banded_L(100, 5))),
                                                         lambda in (0.0, 1e-8, 1.0)
        nlp = VecchiaModel(pattern, samples; lambda)
        ref = madnlp(nlp; print_level=MadNLP.ERROR)
        res = madnlp(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        @test norm(res.solution - ref.solution, Inf) ≤ 1e-6
    end

    @testset "Singular blocks" begin
        # Fewer samples than nonzeros per column: the blocks Hⱼ are singular.
        nlp = VecchiaModel(banded_U(50, 10), gensamples(50, 5); lvar_diag=fill(1e-2, 50), uvar_diag=fill(1e2, 50))
        ref = madnlp(nlp; print_level=MadNLP.ERROR)
        res = madnlp(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        @test res.status == ref.status
        @test res.objective ≈ ref.objective rtol=1e-6
    end

    @testset "Active bounds" begin
        nlp = VecchiaModel(banded_L(100, 5), samples; lvar_diag=fill(1.0, 100), uvar_diag=fill(2.0, 100))
        ref = madnlp(nlp; print_level=MadNLP.ERROR)
        res = madnlp(nlp; kkt_system=VecchiaKKTSystem, print_level=MadNLP.ERROR)
        @test res.status == MadNLP.SOLVE_SUCCEEDED
        @test norm(res.solution - ref.solution, Inf) ≤ 1e-6
    end
end
