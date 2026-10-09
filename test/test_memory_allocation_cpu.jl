@testset "CPU_memory_allocations" begin
    samples = gensamples(100,75)
    U       = banded_U(100,5)
    model   = VecchiaModel(U, samples)
    mems    = NLPModelsTest.test_allocs_nlpmodels(model)

    @test mems[:obj] == 0.0
    @test mems[:grad!] == 0.0
    @test mems[:cons!] == 0.0
    @test mems[:hess_structure!] == 0.0
    @test mems[:jac_structure!] == 0.0
    @test mems[:jac_coord!] == 0.0
    @test mems[:hess_coord!] == 0.0
    @test mems[:hess_lag_coord!] == 0.0
    @test mems[:hprod!] == 0.0
    @test mems[:hprod_lag!] == 0.0
    @test mems[:jprod!] == 0.0
    @test mems[:jtprod!] == 0.0
end
