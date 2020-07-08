"""
    
"""
struct OILMM{
    Tfs<:AbstractVector{<:AbstractGP},
    TU<:AbstractMatrix{<:Real},
    TS<:Diagonal{<:Real},
    Tσ²<:Real,
    TD<:Diagonal{<:Real},
} <: AbstractGP
    fs::Tfs
    U::TU
    S::TS
    σ²::Tσ²
    D::TD
end

(oilmm::OILMM)(x::AbstractVector) = FiniteOILMM(oilmm, x)

"""
    
"""
struct FiniteOILMM{
    Toilmm<:OILMM,
    Tx<:AbstractVector,
} <: ContinuousMultivariateDistribution
    oilmm::Toilmm
    x::Tx
end

function unpack(fx::FiniteOILMM)
    fs = fx.oilmm.fs
    S = fx.oilmm.S
    U = fx.oilmm.U
    D = fx.oilmm.D
    σ² = fx.oilmm.σ²
    x = fx.x
    return fs, S, U, D, σ², x
end

Base.length(f::FiniteOILMM) = length(f.x)

# Note that `cholesky` exploits the diagonal structure of `S`.
function project_data(S::Diagonal, U::AbstractMatrix, Y::AbstractMatrix)
    A = cholesky(S).U \ U'
    return A * Y
end

"""
    rand_latent(rng::AbstractRNG, fx::FiniteOILMM)

Sample from the latent (noiseless) process.

See also `rand`.
"""
function rand_latent(rng::AbstractRNG, fx::FiniteOILMM)
    fs, S, U, D, σ², x = unpack(fx)

    # Generate from the latent processes.
    X = hcat(map((f, d) -> rand(rng, f(x, d)), fs, D.diag)...)

    # Transform latents into observed space.
    return ColVecs(U * cholesky(S).U * X')
end

"""
    rand(rng::AbstractRNG, fx::FiniteOILMM)

Sample from the OILMM, including the observation noise. Follows generative structure of
model 2 from [1].

See also `rand_latent`.

[1] - Bruinsma et al 2020.
"""
function Stheno.rand(rng::AbstractRNG, fx::FiniteOILMM)

    # Sample from the latent process.
    F = rand_latent(rng, fx)

    # Generate iid noise and add to each output.
    return ColVecs(F.X .+ sqrt(fx.oilmm.σ²) .* randn(rng, size(F.X)))
end

"""
    denoised_marginals(fx::FiniteOILMM)

Returns the marginal distribution over the OILMM without the IID noise components.

See also `marginals`.
"""
function denoised_marginals(fx::FiniteOILMM)
    fs, S, U, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = hcat(map(f -> marginals(f(x)), fs)...)
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # Compute the latent -> observed transform.
    H = U * cholesky(S).U

    # Compute the means.
    M = H * M_latent

    # Compute the variances.
    V = abs2.(H) * V_latent

    # Package everything into independent Normal distributions.
    return ColVecs(Normal.(M, sqrt.(V)))
end

"""
    marginals(fx::FiniteOILMM)

Returns the marginal distribution over the output of the OILMM, including observation noise.

See also `denoised_marginals`.
"""
function Stheno.marginals(fx::FiniteOILMM)
    fs, S, U, D, σ², x = unpack(fx)

    # Compute the marginals over the independent latents.
    fs_marginals = hcat(map(f -> marginals(f(x)), fs)...)
    M_latent = mean.(fs_marginals)'
    V_latent = var.(fs_marginals)'

    # Compute the latent -> observed transform.
    H = U * cholesky(S).U

    # Compute the means.
    M = H * M_latent

    # Compute the variances.
    V = abs2.(H) * (V_latent .+ D.diag) .+ σ²

    # Package everything into independent Normal distributions.
    return ColVecs(Normal.(M, sqrt.(V)))
end

"""
    logpdf(fx::FiniteOILMM, Y::ColVecs{<:Real})

Follows implementation in appendix A.4 from Bruinsma et al 2020.
"""
function Stheno.logpdf(fx::FiniteOILMM, Y::ColVecs{<:Real})
    fs, S, U, D, σ², x = unpack(fx)

    # Projection step.
    Yproj = project_data(S, U, Y.X)

    # Latent process log marginal likelihood calculation.
    ΣT = σ² * inv(S) + D
    y_rows = collect(eachrow(Yproj))
    lmls_latents = map((f, s, y) -> logpdf(f(x, s), y), fs, ΣT.diag, y_rows)

    # Reconstruction step.
    p = size(Y.X, 1)
    m = size(Yproj, 1)
    n = length(Y)
    regulariser = -(n * (logdet(cholesky(S)) + (p - m) * log(2π * σ²)) +
        sum(abs2, (I - U * U') * Y.X) / σ²) / 2

    return regulariser + sum(lmls_latents)
end

"""
    posterior(fx::FiniteOILMM, Y::ColVecs{<:Real})

Returns the new `OILMM` object that results from conditioning `fx` on observations `Y`.
"""
function posterior(fx::FiniteOILMM, Y::ColVecs{<:Real})
    fs, S, U, D, σ², x = unpack(fx)

    # Projection step.
    Yproj = project_data(S, U, Y.X)

    # Condition each latent process on the projected observations.
    ΣT = σ² * inv(S) + D
    fs_posterior = map((f, s, y) -> f | Obs(f(x, s), collect(y)), fs, ΣT.diag, eachrow(Yproj))

    return OILMM(fs_posterior, U, S, σ², D)
end
