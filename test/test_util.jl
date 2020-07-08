"""
    consistency_tests(
        rng::AbstractRNG,
        f::OILMM,
        f_naive::Union{AbstractGP, Vector{<:AbstractGP}},
        x::AbstractVector,
        Y::ColVecs{<:Real},
    )

Verify that an OILMM `f` is self-consistent and consistent with the naive GP implementation
`f_naive`.
"""
function consistency_tests(
    rng::AbstractRNG,
    f::OILMM,
    fs_naive::Union{AbstractGP, Vector{<:AbstractGP}},
    x::AbstractVector,
    Y::ColVecs{<:Real},
)
    @assert size(Y.X, 1) == length(fs_naive)

    # Construct train / test sets.
    idx = randperm(rng, length(x))
    Ntr = div(length(x), 2)
    tr_idx = idx[1:Ntr]
    te_idx = idx[Ntr + 1:end]
    xtr = x[tr_idx]
    xte = x[te_idx]
    Ytr = ColVecs(Y.X[:, tr_idx])
    Yte = ColVecs(Y.X[:, te_idx])

    σ² = f.σ²

    # Check log p(Ytr) agrees with naive implementation.
    ys_naive = Vector.(eachrow(Y.X))
    fxs_naive = [f(x, σ²) for f in fs_naive]
    @test logpdf(f(x), Y) ≈ logpdf(fxs_naive, ys_naive)

    # Check that prior marginals agree with naive implementation.
    oilmm_marginals = vec(marginals(f(x)).X)
    naive_marginals = vec(vcat(reshape.(marginals.(fxs_naive), 1, :)...))
    @test mean.(oilmm_marginals) ≈ mean.(naive_marginals)
    @test std.(oilmm_marginals) ≈ std.(naive_marginals)

    # Check that noise-free prior marginals agree with the naive implementation.
    oilmm_marginals = vec(denoised_marginals(f(x)).X)
    fxs_naive_n = [f(x) for f in fs_naive]
    naive_marginals = vec(vcat(reshape.(marginals.(fxs_naive_n), 1, :)...))
    @test mean.(oilmm_marginals) ≈ mean.(naive_marginals)
    @test std.(oilmm_marginals) ≈ std.(naive_marginals)

    # Check that both versions of rand produce samples. This is not a correctness test.
    @test rand_latent(rng, f(x)) isa ColVecs
    @test rand(rng, f(x)) isa ColVecs

    # Ensure that log p(Yte | Ytr) is self-consistent.
    f_posterior = posterior(f(xtr), Ytr)
    x_all = vcat(xtr, xte)
    Y_all = ColVecs(hcat(Ytr.X, Yte.X))
    @test logpdf(f_posterior(xte), Yte) ≈ logpdf(f(x_all), Y_all) - logpdf(f(xtr), Ytr)

    # Construct posterior naively using Stheno.
    ytrs_naive = Vector.(eachrow(Ytr.X))

    fs_naive_posterior = |(
        (fs_naive..., ),
        (Obs.([f(xtr, σ²) for f in fs_naive], ytrs_naive)..., ),
    )
    fxs_naive_posterior = [f(xtr, σ²) for f in fs_naive_posterior]

    # Compare posterior marginals.
    oilmm_posterior_marginals = marginals(f_posterior(xtr))
    naive_posterior_marginals = vec(vcat(reshape.(marginals.(fxs_naive_posterior), 1, :)...))

    @test vec(mean.(oilmm_posterior_marginals.X)) ≈ mean.(naive_posterior_marginals)
    @test vec(std.(oilmm_posterior_marginals.X)) ≈ std.(naive_posterior_marginals)

    # Check that the gradient w.r.t. the logpdf and be computed w.r.t. the observations.
    lml_zygote, pb = Zygote.pullback((f, x, Y) -> logpdf(f(x), Y), f, x, Y)
    @test lml_zygote ≈ logpdf(f(x), Y)

    cotangents_fd = FiniteDifferences.j′vp(
        central_fdm(5, 1),
        (x, Y) -> logpdf(f(x), Y),
        1.0,
        x,
        Y,
    )
    cotangents_ad = pb(1.0)

    @test cotangents_fd[1] ≈ cotangents_ad[2]
    @test cotangents_fd[2].X ≈ cotangents_ad[3].X
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
