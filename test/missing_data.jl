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
        reg = OILMMs.regulariser(S, U, σ², y_missing)
        @test reg ≈ OILMMs.regulariser(S, U, σ², y)

        @testset "OILMMs.project AD" begin

            # Perform forwards-pass and construct pullback.
            (Yproj_ad, ΣT_ad), project_pb = Zygote.pullback(
                OILMMs.project, S, U, y_missing, σ², D,
            )

            # Ensure that the forwards-pass is consistent with usual evaluation.
            @test Yproj_ad ≈ Yproj_missing
            @test ΣT_ad ≈ ΣT_missing

            # Estimate / evaluate cotangent of inputs.
            ΔYproj = randn(size(Yproj_ad))
            ΔΣT_proj = randn(size(ΣT_ad))
            Δout = (ΔYproj, ΔΣT_proj)
            dX_fd = FiniteDifferences.j′vp(
                central_fdm(5, 1),
                (S, U, σ², D) -> OILMMs.project(S, U, y_missing, σ², D),
                Δout, S, U, σ², D,
            )
            dX_ad = project_pb(Δout)

            # Check for (approximate) equality beteen AD and finite differencing.
            @test dX_fd[1] ≈ dX_ad[1]
            @test dX_fd[2] ≈ dX_ad[2]
            @test dX_fd[3] ≈ dX_ad[4]
            @test dX_fd[4] ≈ dX_ad[5]
        end

        @testset "OILMMs.regulariser AD" begin

            # Perform forwards-pass and construct pullback.
            reg_ad, regulariser_ad = Zygote.pullback(OILMMs.regulariser, S, U, σ², y_missing)

            # Ensure that the forwards-pass is consistent with usual evaluation.
            @test reg_ad ≈ reg

            # Estimate / evaluate cotangent of inputs.
            Δreg = randn()
            dX_fd = FiniteDifferences.j′vp(
                central_fdm(5, 1),
                (S, U, σ²) -> OILMMs.regulariser(S, U, σ², y_missing),
                Δreg, S, U, σ²,
            )
            dX_ad = regulariser_ad(Δreg)

            # Check for (approximate) equality beteen AD and finite differencing.
            @test dX_fd[1] ≈ dX_ad[1]
            @test dX_fd[2] ≈ dX_ad[2]
            @test dX_fd[3] ≈ dX_ad[3]
        end
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
