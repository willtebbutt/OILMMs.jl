using Distributions
using FiniteDifferences
using LinearAlgebra
using OILMMs
using Random
using Stheno
using Test
using Zygote

using Stheno: AbstractGP

@testset "OILMMs.jl" begin
    include("test_util.jl")
    include("util.jl")
    include("oilmm.jl")
end
