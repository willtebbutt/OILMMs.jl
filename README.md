# OILMMs.jl: Orthogonal Instantaneous Linear Mixing Models in Julia

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://willtebbutt.github.io/OILMMs.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://willtebbutt.github.io/OILMMs.jl/dev) -->
[![Build Status](https://travis-ci.com/willtebbutt/OILMMs.jl.svg?branch=master)](https://travis-ci.com/willtebbutt/OILMMs.jl)
[![Codecov](https://codecov.io/gh/willtebbutt/OILMMs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/OILMMs.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

An implementation of the Orthogonal Instantaneous Linear Mixing Model (OILMM).

The Python implementation can be found [here](https://github.com/wesselb/oilmm).

## Examples

Please refer to the examples directory for basic usage, or below for a very quick intro.

## API

The API broadly follows [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/)'s.
```julia
f = OILMM(...)
```
constructs an Orthogonal Instantaneous Linear Mixing Model. This object represents a distribution over vector-valued functions -- see the docstring for more info.

```julia
f(x)
```
represents `f` at the input locations `x`.
```julia
logpdf(f(x), y) # compute the log marginal probability of `y` under the model.
rand(rng, f(x)) # sample from `f` at `x`, for random number generator `rng`.
marginals(f(x)) # compute the marginal statistics of `f` at `x`.
```
`y` should be an `AbstractVector{<:Real}` of the same length as `x`.

To perform inference, simply invoke the `posterior` function:
```
f_post = posterior(f(x), y)
```
`f_post` is then another `OILMM` that is the posterior distribution. That this works is one of the very convenient properties of the OILMM.

All public functions should have docstrings. If you encounter something that is unclear, please raise an issue so that it can be fixed.

## Worked Example

```julia
using AbstractGPs
using LinearAlgebra
using OILMMs
using Random

# Specify and construct an OILMM.
p = 10
m = 3
U, s, _ = svd(randn(p, m))
σ² = 0.1

f = OILMM(
    [GP(SEKernel()) for _ in 1:m],
    U,
    Diagonal(s),
    Diagonal(rand(m) .+ 0.1),
);

# Sample from the model.
x = MOInput(randn(10), p);
fx = f(x, σ²);

rng = MersenneTwister(123456);
y = rand(rng, fx);

# Compute the logpdf of the data under the model.
logpdf(fx, y)

# Perform posterior inference. This produces another OILMM.
f_post = posterior(fx, y)

# Compute the posterior marginals. We can also use `rand` and `logpdf` as before.
post_marginals = marginals(f_post(x));
```

## Worked Example using TemporalGPs.jl.

[TemporalGPs.jl](https://github.com/willtebbutt/TemporalGPs.jl/) makes inference and learning in GPs for time series much more efficient than performing exact inference.
It plays nicely with this package, and can be used to accelerate inference in an OILMM
simply by wrapping each of the base processes using `to_sde`. See the TemporalGPs.jl docs
for more info on this.

```julia
using AbstractGPs
using LinearAlgebra
using OILMMs
using Random
using TemporalGPs

# Specify and construct an OILMM.
p = 10
m = 3
U, s, _ = svd(randn(p, m))
σ² = 0.1

f = OILMM(
    [to_sde(GP(Matern52Kernel()), SArrayStorage(Float64)) for _ in 1:m],
    U,
    Diagonal(s),
    Diagonal(rand(m) .+ 0.1),
);

# Sample from the model. LARGE DATA SET!
x = MOInput(RegularSpacing(0.0, 1.0, 1_000_000), p);
fx = f(x, σ²);
rng = MersenneTwister(123456);
y = rand(rng, fx);

# Compute the logpdf of the data under the model.
logpdf(fx, y)

# Perform posterior inference. This produces another OILMM.
f_post = OILMMs.posterior(fx, y)

# Compute the posterior marginals. We can also use `rand` and `logpdf` as before.
post_marginals = marginals(f_post(x));
```


## Bib Info
Please refer to the CITATION.bib file.
