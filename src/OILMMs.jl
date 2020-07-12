module OILMMs

using Distributions
using LinearAlgebra
using Random
using Stheno
using Zygote
using ZygoteRules

using Stheno: AbstractGP, ColVecs

export OILMM, posterior

include("util.jl")
include("oilmm.jl")
include("missing_data.jl")

end # module
