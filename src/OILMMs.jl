module OILMMs

using Distributions
using LinearAlgebra
using Random
using Stheno

using Stheno: AbstractGP, ColVecs

export OILMM, posterior, denoised_marginals, rand_latent

include("oilmm.jl")

end # module
