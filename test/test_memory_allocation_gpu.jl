function test_allocs_nlpmodels_gpu(nlp::AbstractNLPModel; linear_api = false, exclude = [])
  nlp_allocations = Dict(
    :obj => NaN,
    :grad! => NaN,
    :hess_structure! => NaN,
    :hess_coord! => NaN,
    :hprod! => NaN,
    :hess_op_prod! => NaN,
    :cons! => NaN,
    :jac_structure! => NaN,
    :jac_coord! => NaN,
    :jprod! => NaN,
    :jtprod! => NaN,
    :jac_op_prod! => NaN,
    :jac_op_transpose_prod! => NaN,
    :hess_lag_coord! => NaN,
    :hprod_lag! => NaN,
    :hess_lag_op! => NaN,
    :hess_lag_op_prod! => NaN,
  )

  if !(obj in exclude)
    x = get_x0(nlp)
    obj(nlp, x)
    nlp_allocations[:obj] = CUDA.@allocated obj(nlp, x)
  end
  if !(grad in exclude)
    x = get_x0(nlp)
    g = similar(x)
    grad!(nlp, x, g)
    nlp_allocations[:grad!] = CUDA.@allocated grad!(nlp, x, g)
  end
  if !(hess in exclude)
    rows = CuVector{Int}(undef, nlp.meta.nnzh)
    cols = CuVector{Int}(undef, nlp.meta.nnzh)
    hess_structure!(nlp, rows, cols)
    nlp_allocations[:hess_structure!] = CUDA.@allocated hess_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = CuVector{eltype(x)}(undef, nlp.meta.nnzh)
    hess_coord!(nlp, x, vals)
    nlp_allocations[:hess_coord!] = CUDA.@allocated hess_coord!(nlp, x, vals)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      hess_coord!(nlp, x, y, vals)
      nlp_allocations[:hess_lag_coord!] = CUDA.@allocated hess_coord!(nlp, x, y, vals)
    end
  end
  if !(hprod in exclude)
    x = get_x0(nlp)
    v = copy(x)
    Hv = similar(x)
    hprod!(nlp, x, v, Hv)
    nlp_allocations[:hprod!] = CUDA.@allocated hprod!(nlp, x, v, Hv)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      hprod!(nlp, x, y, v, Hv)
      nlp_allocations[:hprod_lag!] = CUDA.@allocated hprod!(nlp, x, y, v, Hv)
    end
  end
  if !(hess_op in exclude)
    x = get_x0(nlp)
    Hv = similar(x)
    v = copy(x)
    H = hess_op!(nlp, x, Hv)
    mul!(Hv, H, v)
    nlp_allocations[:hess_op_prod!] = CUDA.@allocated mul!(Hv, H, v)
    if get_ncon(nlp) > 0
      y = get_y0(nlp)
      H = hess_op!(nlp, x, y, Hv)
      mul!(Hv, H, v)
      nlp_allocations[:hess_lag_op_prod!] = CUDA.@allocated mul!(Hv, H, v)
    end
  end

  if get_ncon(nlp) > 0 && !(cons in exclude)
    x = get_x0(nlp)
    c = CuVector{eltype(x)}(undef, get_ncon(nlp))
    cons!(nlp, x, c)
    nlp_allocations[:cons!] = CUDA.@allocated cons!(nlp, x, c)
  end
  if get_ncon(nlp) > 0 && !(jac in exclude)
    rows = CuVector{Int}(undef, nlp.meta.nnzj)
    cols = CuVector{Int}(undef, nlp.meta.nnzj)
    jac_structure!(nlp, rows, cols)
    nlp_allocations[:jac_structure!] = CUDA.@allocated jac_structure!(nlp, rows, cols)
    x = get_x0(nlp)
    vals = CuVector{eltype(x)}(undef, nlp.meta.nnzj)
    jac_coord!(nlp, x, vals)
    nlp_allocations[:jac_coord!] = CUDA.@allocated jac_coord!(nlp, x, vals)
  end
  if get_ncon(nlp) > 0 && !(jprod in exclude)
    x = get_x0(nlp)
    v = copy(x)
    Jv = CuVector{eltype(x)}(undef, get_ncon(nlp))
    jprod!(nlp, x, v, Jv)
    nlp_allocations[:jprod!] = CUDA.@allocated jprod!(nlp, x, v, Jv)
  end
  if get_ncon(nlp) > 0 && !(jtprod in exclude)
    x = get_x0(nlp)
    v = copy(get_y0(nlp))
    Jtv = similar(x)
    jtprod!(nlp, x, v, Jtv)
    nlp_allocations[:jtprod!] = CUDA.@allocated jtprod!(nlp, x, v, Jtv)
  end
  if get_ncon(nlp) > 0 && !(jac_op in exclude)
    x = get_x0(nlp)
    Jtv = similar(x)
    Jv = CuVector{eltype(x)}(undef, get_ncon(nlp))

    v = copy(x)
    w = copy(get_y0(nlp))
    J = jac_op!(nlp, x, Jv, Jtv)
    mul!(Jv, J, v)
    nlp_allocations[:jac_op_prod!] = CUDA.@allocated mul!(Jv, J, v)
    mul!(Jtv, J', w)
    nlp_allocations[:jac_op_transpose_prod!] = CUDA.@allocated mul!(Jtv, J', w)
  end

  for type in (:nln, :lin)
    nn = type == :lin ? nlp.meta.nlin : nlp.meta.nnln
    nnzj = type == :lin ? nlp.meta.lin_nnzj : nlp.meta.nln_nnzj
    if !linear_api || (nn == 0)
      continue
    end
    if !(cons in exclude)
      x = get_x0(nlp)
      c = CuVector{eltype(x)}(undef, nn)
      fun = Symbol(:cons_, type, :!)
      eval(fun)(nlp, x, c)
      nlp_allocations[fun] = CUDA.@allocated eval(fun)(nlp, x, c)
    end
    if !(jac in exclude)
      rows = CuVector{Int}(undef, nnzj)
      cols = CuVector{Int}(undef, nnzj)
      fun = type == :lin ? jac_lin_structure! : jac_nln_structure! # eval(fun) would allocate here
      fun(nlp, rows, cols)
      nlp_allocations[Symbol(fun)] = CUDA.@allocated fun(nlp, rows, cols)
      x = get_x0(nlp)
      vals = CuVector{eltype(x)}(undef, nnzj)
      fun = Symbol(:jac_, type, :_coord!)
      eval(fun)(nlp, x, vals)
      nlp_allocations[fun] = CUDA.@allocated eval(fun)(nlp, x, vals)
    end
    if !(jprod in exclude)
      x = get_x0(nlp)
      v = copy(x)
      Jv = CuVector{eltype(x)}(undef, nn)
      fun = Symbol(:jprod_, type, :!)
      eval(fun)(nlp, x, v, Jv)
      nlp_allocations[fun] = CUDA.@allocated eval(fun)(nlp, x, v, Jv)
    end
    if !(jtprod in exclude)
      x = get_x0(nlp)
      v = copy(get_y0(nlp)[1:nn])
      Jtv = similar(x)
      fun = Symbol(:jtprod_, type, :!)
      eval(fun)(nlp, x, v, Jtv)
      nlp_allocations[fun] = CUDA.@allocated eval(fun)(nlp, x, v, Jtv)
    end
    if !(jac_op in exclude)
      x = get_x0(nlp)
      Jtv = similar(x)
      Jv = CuVector{eltype(x)}(undef, nn)

      v = copy(x)
      w = randn(eltype(x), nn)
      fun = Symbol(:jac_, type, :_op!)
      if type == :lin
        J = jac_lin_op!(nlp, x, Jv, Jtv)
        mul!(Jv, J, v)
        nlp_allocations[Symbol(:jac_lin_op_prod!)] = CUDA.@allocated mul!(Jv, J, v)
        mul!(Jtv, J', w)
        nlp_allocations[Symbol(:jac_lin_op_transpose_prod!)] = CUDA.@allocated mul!(Jtv, J', w)
      else
        J = jac_nln_op!(nlp, x, Jv, Jtv)
        mul!(Jv, J, v)
        nlp_allocations[Symbol(:jac_nln_op_prod!)] = CUDA.@allocated mul!(Jv, J, v)
        mul!(Jtv, J', w)
        nlp_allocations[Symbol(:jac_nln_op_transpose_prod!)] = CUDA.@allocated mul!(Jtv, J', w)
      end
    end
  end
  return nlp_allocations
end

@testset "GPU_memory_allocations" begin
    samples = gensamples(100,75)
    L     = banded_L(100,5)
    model = VecchiaModel(L, CuMatrix(samples))
    mems = test_allocs_nlpmodels_gpu(model)

    @test mems[:obj] == 16.0  # these allocations are related to allocations in "sum" and "dot"
    @test mems[:grad!] == 0.0
    @test_broken mems[:cons!] == 0.0
    @test mems[:hess_structure!] == 0.0
    @test mems[:jac_structure!] == 0.0
    @test mems[:jac_coord!] == 0.0
    @test mems[:hess_coord!] == 0.0
    @test mems[:hess_lag_coord!] == 0.0
    @test mems[:hprod!] == 0.0
    @test mems[:hprod_lag!] == 0.0
    # Same allocation as cons!: broadcast with a view indexed by a CuVector
    @test_broken mems[:jprod!] == 0.0
    @test_broken mems[:jtprod!] == 0.0
end
