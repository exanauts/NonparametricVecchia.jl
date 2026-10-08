@testset "Vecchia.jl extension" begin

  samples = gensamples(100, 75)
  pts     = [SVector{1,Float64}(j) for j in 1:100]

  (perm, nlp) = VecchiaModel(pts, permutedims(samples), 
                             RandomOrdering(StableRNG(1234)), 
                             KNNConditioning(10);
                             lvar_diag=fill(inv(sqrt(1.0)),  length(pts)),
                             uvar_diag=fill(inv(sqrt(1e-2)), length(pts)),
                             lambda=1e-3)

  ref = madnlp(nlp; tol=1e-10)
  res = madnlp(nlp; tol=1e-10, kkt_system=VecchiaKKTSystem)
  @test res.status == MadNLP.SOLVE_SUCCEEDED
  @test norm(res.solution - ref.solution, Inf) ≤ 1e-6
end
