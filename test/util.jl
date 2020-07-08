@testset "util" begin
    @testset "eachrow" begin
        X = randn(2, 5)

        y, pb = Zygote.pullback(collect ∘ eachrow, X)
        @test y == collect(eachrow(X))

        # Just make the seed the output since it's a valid cotangent for itself.
        ȳ = y

        # Compute pullback using Zygote and FiniteDifferences and test they roughly agree.
        dX_fd = FiniteDifferences.j′vp(central_fdm(5, 1), collect ∘ eachrow, ȳ, X)
        dX_ad = pb(ȳ)

        @test first(dX_fd) ≈ first(dX_ad)
    end
    @testset "inv(::Diagonal)" begin
        x = randn(3)

        f = diag ∘ inv ∘ Diagonal

        y, pb = Zygote.pullback(f, x)
        @test y == f(x)

        ȳ = randn(3)

        dX_fd = FiniteDifferences.j′vp(central_fdm(5, 1), f, ȳ, x)
        dX_ad = pb(ȳ)

        @test first(dX_fd) ≈ first(dX_ad)
    end
end
