module OILMMs

using Distributions
using LinearAlgebra
using Random
using Stheno
using Zygote
using ZygoteRules

using Stheno: AbstractGP, ColVecs

export OILMM, posterior, denoised_marginals, rand_latent

include("util.jl")
include("oilmm.jl")

end # module
