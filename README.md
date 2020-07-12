# OILMMs.jl: Orthogonal Instantaneous Linear Mixing Models in Julia

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://willtebbutt.github.io/OILMMs.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://willtebbutt.github.io/OILMMs.jl/dev) -->
[![Build Status](https://travis-ci.com/willtebbutt/OILMMs.jl.svg?branch=master)](https://travis-ci.com/willtebbutt/OILMMs.jl)
[![Codecov](https://codecov.io/gh/willtebbutt/OILMMs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/willtebbutt/OILMMs.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

An implementation of the Orthogonal Instantaneous Linear Mixing Model (OILMM).

The Python implementation can be found [here](https://github.com/wesselb/oilmm).

## Examples

Please refer to the examples directory for basic usage.

## API

The API broadly follows [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/)'s.
```
f = OILMM(...)
```
construct an Orthogonal Instantaneous Linear Mixing Model. This object represents a distribution over vector-valued functions -- see the docstring for more info.

```
f(x)
```
represents `f` at the input locations `x`.
```
logpdf(f(x), y) # compute the log marginal probability of `y` under the model.
rand(rng, f(x)) # sample from `f` at `x`, for random number generator `rng`.
marginals(f(x)) # compute the marginal statistics of `f` at `x`.
```
`y` should be a `ColVecs` of the same length as `x`. A `ColVecs` is simply a wrapper around a `N-output x N-observations` matrix that tells this package that you mean to interpret said matrix as such. This object is helpful in that it prevents accidentally getting the observations around the wrong way. See [Stheno.jl](https://github.com/willtebbutt/Stheno.jl/)'s docs for more info.
Additionally, `rand` and `marginals` return `ColVecs` objects. You can query the underlying matrix via the `.X` field.

To perform inference, simply invoke the `posterior` function:
```
f_post = posterior(f(x), y)
```
`f_post` is then another `OILMM` that is the posterior distribution. That this works is one of the very convenient properties of the OILMM.

All public functions should have docstrings. If you encounter something that is unclear, please raise an issue so that it can be fixed.

## Bib Info
Please refer to the CITATION.bib file.
