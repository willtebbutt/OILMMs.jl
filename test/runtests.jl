using Distributions
using FiniteDifferences
using LinearAlgebra
using OILMMs
using Random
using Stheno
using Test
using Zygote

using OILMMs: denoised_marginals, rand_latent
using Stheno: AbstractGP

# Helper functionality, doesn't actually run any tests.
include("test_util.jl")

@testset "OILMMs.jl" begin
    include("util.jl")
    include("oilmm.jl")
    include("missing_data.jl")
end
