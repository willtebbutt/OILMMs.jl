using AbstractGPs
using Distributions
using FiniteDifferences
using LinearAlgebra
using OILMMs
using Random
using Stheno
using Test
using Zygote

using AbstractGPs: AbstractGP, FiniteGP
using OILMMs: denoised_marginals, rand_latent
using Stheno: GaussianProcessProbabilisticProgramme

# Helper functionality, doesn't actually run any tests.
include("test_util.jl")

@testset "OILMMs.jl" begin
    include("util.jl")
    include("oilmm.jl")
    include("missing_data.jl")
end
