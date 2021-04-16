module OILMMs

using AbstractGPs
using ChainRulesCore
using Distributions
using FillArrays
using KernelFunctions
using LinearAlgebra
using Random
using Zygote
using ZygoteRules

using AbstractGPs: AbstractGP, FiniteGP
using KernelFunctions: MOInput


include("util.jl")
include("oilmm.jl")
include("missing_data.jl")

export OILMM

end # module
