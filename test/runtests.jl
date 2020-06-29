using Distributions
using LinearAlgebra
using OILMMs
using Random
using Stheno
using Test

using Stheno: AbstractGP

@testset "OILMMs.jl" begin
    include("test_util.jl")
    include("oilmm.jl")
end
