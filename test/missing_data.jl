@testset "missing_data" begin

    @testset "nothing actually missing" begin
        # Construct a toy data set with no missing data.
        Y = randn(2, 5)
        y = ColVecs(Y)

        # Copy the toy data set, but change the type so that it _might_ have missing data.
        Y_missing = Matrix{Union{Missing, Float64}}(undef, size(Y))
        copy!(Y_missing, Y)
        y_missing = ColVecs(Y_missing)

        # Construct a random projection.
        U, s, _ = svd(randn(2, 1))
        S = Diagonal(s)

        # Construct noise model.
        σ² = 0.93
        D = Diagonal(abs.(randn(size(S, 1))) .+ 1e-2)

        # Ensure that both projection operations produce the same result.
        Yproj, ΣT = OILMMs.project(S, U, y, σ², D)
        Yproj_missing, ΣT_missing = OILMMs.project(S, U, y_missing, σ², D)
        @test Yproj ≈ Yproj_missing
        @test ΣT ≈ ΣT_missing

        # Ensure that the regularisation terms agree.
        @test OILMMs.regulariser(S, U, σ², y) ≈ OILMMs.regulariser(S, U, σ², y_missing)
    end

    @testset "identity projection" begin

        # Construct a toy data set and make some bits of it missing.
        Y = Matrix{Union{Missing, Float64}}(undef, 3, 4)
        Y .= randn(3, 4)
        Y[1, 2] = missing
        Y[2, 1] = missing
        y = ColVecs(Y)

        # Compute the output of the project. The missings are just zeroed.
        Yproj = deepcopy(Y)
        Yproj[1, 2] = 0.0
        Yproj[2, 1] = 0.0

        # Construct the identity transformation.
        U, s, _ = svd(Matrix{Float64}(I, 3, 3))
        S = Diagonal(s)

        # Construct noise model.
        σ² = 0.93
        D = Diagonal(abs.(randn(size(Y, 1))) .+ 1e-2)

        # Ensure that the projection operation recovers the zeroed data.
        @test OILMMs.project(S, U, y, σ², D)[1] == Yproj
    end
end
