using LinearAlgebra
using OILMMs
using Optim
using Plots
using Random
using Stheno
using Zygote

using Zygote: gradient

# Fix a seed.
rng = MersenneTwister(123456)

# Construct a simple OILMM.
dim_observed = 25
dim_latent = 3
U, s, _ = svd(randn(rng, dim_observed, dim_latent))
σ² = 0.1
D = Diagonal(fill(1e-4, dim_latent))
fs = [GP(stretch(EQ(), λ), GPC()) for λ in abs.(randn(dim_latent)) .+ 0.1]
f = OILMM(fs, U, Diagonal(s), σ², D)

# Sample synthetic data from the prior.
N_train = 150
N_test = 350
x = range(0.0, 10.0; length=N_train)
x_test = range(-5.0, 15.0; length=N_test)
y = rand(rng, f(x))

# Perform inference using the true model, generate samples, compute marginal statistics.
f_post = posterior(f(x), y)
fs_post = rand_latent(rng, f_post(x_test))
marginals_post = marginals(f_post(x_test))

# Plot the first couple of posterior processes.
# We'll display them later once we've done learning-related stuff below.
true_post_colour = :blue
first_marginals = marginals_post.X[1, :];
true_posterior_plot = plot();
plot!(true_posterior_plot, x_test, fs_post.X[1, :];
    linewidth=1, linealpha=0.7, color=true_post_colour, label="",
);
plot!(true_posterior_plot, x_test, [mean.(first_marginals) mean.(first_marginals)];
    linewidth=0.0,
    color=true_post_colour,
    fillrange=[quantile.(first_marginals, 0.01), quantile.(first_marginals, 0.99)],
    fillalpha=0.3,
    label="",
);
plot!(true_posterior_plot, x_test, mean.(first_marginals);
    linewidth=2, linealpha=1, color=true_post_colour, label="posterior (true params)",
);
scatter!(true_posterior_plot, x, y.X[1, :];
    color=:black, label="",
);
# display(true_posterior_plot); # uncomment to plot immediately.

# Learning the `H` matrix from data. This is a little verbose, but it does the job.

# Ideally this function would be automatically handled. At the time of writing this example,
# this is unfortunately not the case. It feels really silly that I still have to write this
# function.
function unpack(θ::Vector)

    # Check that everything is conformal.
    length_H = dim_observed * dim_latent
    length_d = dim_latent
    length_λs = dim_latent
    @assert length(θ) == length_H + 1 + length_d + length_λs

    # Unpack θ and get parameters of the OILMM.
    H = reshape(θ[1:length_H], dim_observed, dim_latent)
    U, s = svd(H)
    S = Diagonal(s)

    σ² = exp(θ[length_H + 1]) + 1e-3

    # Note that fixing the noise on the latent processes seems to stabilise learning.
    # You could actually learn these parameters of course.
    d = fill(1e-4, length_d)
    D = Diagonal(d)

    λs = exp.(θ[length_H + 1 + length_d + 1:end])

    return U, S, σ², D, λs
end

function build_model(θ::Vector)

    # Get the parameters.
    U, S, σ², D, λs = unpack(θ)

    # Build the latent GPs.
    fs = map(λ -> GP(stretch(EQ(), λ), GPC()), λs)

    # Construct the OILMM.
    return OILMM(fs, U, S, σ², D)
end

# Regularise the nlml very slightly for stability -- best not thought of as a diffuse prior.
nlml(θ::Vector) = -logpdf(build_model(θ)(x), y) + 1e-3 * sum(abs2, θ)

θ0 = randn(rng, dim_observed * dim_latent + 1 + dim_latent + dim_latent);
opts = Optim.Options(show_trace=true, iterations=300)
results = Optim.optimize(nlml, θ->gradient(nlml, θ)[1], θ0, BFGS(), opts; inplace=false)
f_ml = build_model(results.minimizer);

# Do inference at the parameters maximising the marginal likelihood.
f_post_ml = posterior(f_ml(x), y)
fs_post_ml = rand_latent(rng, f_post_ml(x_test))
marginals_post_ml = marginals(f_post_ml(x_test))

# Visualise the posterior at the parameters maximising the marginal likelihood.
ml_post_colour = :red
ml_first_marginals = marginals_post_ml.X[1, :];
plot!(true_posterior_plot, x_test, fs_post_ml.X[1, :];
    linewidth=1, linealpha=0.7, color=ml_post_colour, label="",
);
plot!(true_posterior_plot, x_test, [mean.(ml_first_marginals) mean.(ml_first_marginals)];
    linewidth=0.0,
    color=ml_post_colour,
    fillrange=[quantile.(ml_first_marginals, 0.01), quantile.(ml_first_marginals, 0.99)],
    fillalpha=0.3,
    label="",
);
plot!(true_posterior_plot, x_test, mean.(ml_first_marginals);
    linewidth=2, linealpha=1, color=ml_post_colour, label="posterior (learned params)",
);
display(true_posterior_plot);
