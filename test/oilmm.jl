@testset "oilmm" begin
    @testset "single output" begin
        rng = MersenneTwister(123456)

        # Specify GP.
        a = randn(rng)
        f = a * GP(Matern52(), GPC())

        # Specify equivalent OILMM.
        fs = [GP(Matern52(), GPC())]
        U = reshape([1.0], 1, 1)
        S = Diagonal([abs2(a)])
        σ²_n = 0.12
        D = Diagonal([0.0])
        oilmm = OILMM(fs, U, S, σ²_n, D)

        # Specify inputs and generate observations.
        x = collect(range(-3.37, 3.12; length=11))
        y = rand(rng, f(x, σ²_n))
        Y = ColVecs(reshape(y, 1, :))

        # Check that the posteriors agree.
        consistency_tests(rng, oilmm, f, x, Y)
    end
    @testset "P independent processes" begin
        rng = MersenneTwister(123456)
        P = 11
        N = 15

        # Specify a collection of GPs
        as = randn(rng, P)
        gpc = GPC()
        fs = [as[p] * GP(Matern52(), gpc) for p in 1:P]

        # Specify equivalent OILMM.
        U = collect(Diagonal(ones(P)))
        S = Diagonal(abs2.(as))
        σ² = 0.15
        D = Diagonal(zeros(P))
        oilmm = OILMM([GP(Matern52(), GPC()) for p in 1:P], U, S, σ², D)

        # Specify inputs and generate observations.
        x = collect(range(-5.43, 1.76; length=N))
        fxs = [f(x, σ²) for f in fs]
        ys = rand(rng, fxs)
        Y = ColVecs(collect(hcat(ys...)'))

        consistency_tests(rng, oilmm, fs, x, Y)
    end
    @testset "Full Rank, Dense H" begin
        rng = MersenneTwister(123456)
        P = 3
        N = 15

        # Construct a random orthogonal H.
        U, S_diag, _ = svd(randn(rng, P, P))
        H = U * Diagonal(sqrt.(S_diag))

        # Specify a collection of GPs
        gpc = GPC()
        zs = [GP(Matern52(), gpc) for p in 1:P]
        fs = [sum(H[p, :] .* zs) for p in 1:P]

        # Specify equivalent OILMM.
        σ² = 0.11
        D = Diagonal(zeros(P))
        oilmm = OILMM(zs, U, Diagonal(S_diag), σ², D)

        # Specify inputs and generate observations.
        x = collect(range(-5.43, 1.76; length=N))

        fxs = [f(x, σ²) for f in fs]
        ys = rand(rng, fxs)
        Y = ColVecs(collect(hcat(ys...)'))

        consistency_tests(rng, oilmm, fs, x, Y)
    end
end
