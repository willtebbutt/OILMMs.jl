function Base.vcat(x::MOInput, y::MOInput)
    x.out_dim == y.out_dim || throw(error("out_dim mismatch"))
    return MOInput(vcat(x.x, y.x), x.out_dim)
end

@testset "oilmm" begin
    @testset "single output" begin
        rng = MersenneTwister(123456)

        # Specify inputs and generate observations.
        x = collect(range(-3.37, 3.12; length=11))
        tr_idx = randperm(rng, length(x))[1:6]
        te_idx = setdiff(eachindex(x), tr_idx)
        x_tr_raw = x[tr_idx]
        x_te_raw = x[te_idx]

        # Noise variance.
        σ² = 0.12

        # Specify the equivalent GPPP.
        a = randn(rng)
        f_naive = @gppp let
            f = a * GP(Matern52Kernel())
        end

        # Specify equivalent OILMM.
        fs = [GP(Matern52Kernel())]
        U = reshape([1.0], 1, 1)
        S = Diagonal([abs2(a)])
        D = Diagonal([0.0])
        f = OILMM(fs, U, S, D)
        x_tr = MOInput(x_tr_raw, 1)
        x_te = MOInput(x_te_raw, 1)
        y_tr = rand(rng, f(x_tr, σ²))
        y_te = rand(rng, f(x_te, σ²))

        # Check that the posteriors agree.
        consistency_tests(
            rng, f, f_naive;
            x_tr=x_tr,
            x_te=x_te,
            x_naive_tr=GPPPInput(:f, x_tr_raw),
            x_naive_te=GPPPInput(:f, x_te_raw),
            y_tr=y_tr,
            y_te=y_te,
            σ²=σ²,
        )
    end
    # @testset "P independent processes" begin
    #     rng = MersenneTwister(123456)
    #     P = 11
    #     N = 15

    #     # Specify a collection of GPs
    #     as = randn(rng, P)
    #     fs = [as[p] * GP(Matern52Kernel()) for p in 1:P]

    #     # Specify equivalent OILMM.
    #     U = collect(Diagonal(ones(P)))
    #     S = Diagonal(abs2.(as))
    #     σ² = 0.15
    #     D = Diagonal(zeros(P))
    #     oilmm = OILMM([GP(Matern52Kernel()) for p in 1:P], U, S, σ², D)

    #     # Specify inputs and generate observations.
    #     x = collect(range(-5.43, 1.76; length=N))
    #     fxs = [f(x, σ²) for f in fs]
    #     ys = rand(rng, fxs)
    #     Y = ColVecs(collect(hcat(ys...)'))

    #     consistency_tests(rng, oilmm, fs, x, Y)
    # end
    # @testset "Full Rank, Dense H" begin
    #     rng = MersenneTwister(123456)
    #     P = 3
    #     N = 15

    #     # Construct a random orthogonal H.
    #     U, S_diag, _ = svd(randn(rng, P, P))
    #     H = U * Diagonal(sqrt.(S_diag))

    #     # Specify a collection of GPs
    #     zs = [GP(Matern52Kernel()) for p in 1:P]
    #     fs = [sum(H[p, :] .* zs) for p in 1:P]

    #     # Specify equivalent OILMM.
    #     σ² = 0.11
    #     D = Diagonal(zeros(P))
    #     oilmm = OILMM(zs, U, Diagonal(S_diag), σ², D)

    #     # Specify inputs and generate observations.
    #     x = collect(range(-5.43, 1.76; length=N))

    #     fxs = [f(x, σ²) for f in fs]
    #     ys = rand(rng, fxs)
    #     Y = ColVecs(collect(hcat(ys...)'))

    #     consistency_tests(rng, oilmm, fs, x, Y)
    # end
end
