"""
    consistency_tests(
        rng::AbstractRNG,
        f::OILMM,
        f_naive::GaussianProcessProbabilisticProgramme,
        x::AbstractVector,
        Y::ColVecs{<:Real},
    )

Verify that an OILMM `f` is self-consistent and consistent with the naive GP implementation
`f_naive`.
"""
function consistency_tests(
    rng::AbstractRNG,
    f::OILMM,
    f_naive::GaussianProcessProbabilisticProgramme;
    x_tr::AbstractVector,
    x_te::AbstractVector,
    x_naive_tr::AbstractVector,
    x_naive_te::AbstractVector,
    y_tr::AbstractVector,
    y_te::AbstractVector,
    σ²::Real,
)
    # Set up the finite-dimensional marginals.
    fx_tr = f(x_tr, σ²)
    fx_tr_naive = f_naive(x_naive_tr, σ²)

    # Check log p(Ytr) agrees with naive implementation. 
    @test logpdf(fx_tr, y_tr) ≈ logpdf(fx_tr_naive, y_tr)

    # Check that prior marginals agree with naive implementation.
    @test mean(fx_tr) ≈ mean(fx_tr_naive)
    @test var(fx_tr) ≈ var(fx_tr_naive)

    # Check that noise-free prior marginals agree with the naive implementation.
    oilmm_marginals = denoised_marginals(f(x_tr))
    naive_marginals = marginals(f_naive(x_naive_tr))
    @test mean.(oilmm_marginals) ≈ mean.(naive_marginals)
    @test std.(oilmm_marginals) ≈ std.(naive_marginals)

    # Check that both versions of rand produce samples. This is not a correctness test.
    @test rand_latent(rng, f(x_tr)) isa Vector{<:Real}
    @test rand(rng, f(x_tr)) isa Vector{<:Real}

    # Ensure that log p(Yte | Ytr) is self-consistent.
    f_posterior = posterior(f(x_tr, σ²), y_tr)
    x_all = vcat(x_tr, x_te)
    y_all = vcat(y_tr, y_te)
    @test logpdf(f_posterior(x_te, σ²), y_te) ≈
        logpdf(f(x_all, σ²), y_all) - logpdf(f(x_tr, σ²), y_tr)

    # Construct the posterior naively using Stheno.
    f_posterior = posterior(f(x_tr, σ²), y_tr)
    f_posterior_naive = posterior(f_naive(x_naive_tr, σ²), y_tr)

    # Compare posterior marginals.
    @test mean(f_posterior(x_tr, σ²)) ≈ mean(f_posterior_naive(x_naive_tr, σ²))
    @test var(f_posterior(x_tr)) ≈ var(f_posterior_naive(x_naive_tr))

    @test mean(f_posterior(x_te)) ≈ mean(f_posterior_naive(x_naive_te))
    @test var(f_posterior(x_te)) ≈ var(f_posterior_naive(x_naive_te))

    # Check that the gradient w.r.t. the logpdf and be computed w.r.t. the observations.
    lml_zygote, pb = Zygote.pullback((f, x, y) -> logpdf(f(x, σ²), y), f, x_tr, y_tr)
    @test lml_zygote ≈ logpdf(f(x_tr, σ²), y_tr)

    Δ = randn(rng)
    cotangents_fd = FiniteDifferences.j′vp(
        central_fdm(5, 1),
        y -> logpdf(f(x_tr, σ²), y),
        Δ,
        y_tr,
    )
    cotangents_ad = pb(Δ)

    @test cotangents_fd[1] ≈ cotangents_ad[3]
end

function consistency_tests(
    rng::AbstractRNG,
    f::OILMM,
    f_naive::AbstractGP,
    x::AbstractVector,
    Y::ColVecs{<:Real},
)
    consistency_tests(rng, f, [f_naive], x, Y)
end

function FiniteDifferences.to_vec(X::ColVecs)
    x_vec, back = to_vec(X.X)
    function from_vec_ColVecs(x)
        return ColVecs(back(x))
    end
    return x_vec, from_vec_ColVecs
end
