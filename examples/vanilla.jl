using LinearAlgebra
using OILMMs
using Optim
using Plots
using Random
using Statistics
using Stheno
using Zygote

using Zygote: gradient





#
# Complete Data
#

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
N_train = 150;
N_test = 350;
x = range(0.0, 10.0; length=N_train);
x_test = range(-5.0, 15.0; length=N_test);
y = rand(rng, f(x));

# Utility function to do repeated plotting tasks. Also shows how to do inference.
function extend_posterior_plot!(
    posterior_plot,
    f::OILMM,
    y_train::ColVecs,
    colour::Symbol,
    label::String,
)
    # Perform inference.
    f_post = posterior(f(x), y_train)
    fs_post = OILMMs.rand_latent(rng, f_post(x_test))
    marginals_post = marginals(f_post(x_test))

    # Plot the first couple of posterior processes.
    # We'll display them later once we've done learning-related stuff below.
    first_marginals = marginals_post.X[1, :]
    plot!(posterior_plot, x_test, fs_post.X[1, :];
        linewidth=1, linealpha=0.7, color=colour, label="",
    )
    plot!(posterior_plot, x_test, [mean.(first_marginals) mean.(first_marginals)];
        linewidth=0.0,
        color=colour,
        fillrange=[quantile.(first_marginals, 0.01), quantile.(first_marginals, 0.99)],
        fillalpha=0.3,
        label="",
    )
    plot!(posterior_plot, x_test, mean.(first_marginals);
        linewidth=2, linealpha=1, color=colour, label=label,
    )

    return nothing
end

posterior_plot = plot();
extend_posterior_plot!(posterior_plot, f, y, :blue, "exact, true params)");
scatter!(posterior_plot, x, y.X[1, :];
    color=:black, label="",
);
# display(posterior_plot); # uncomment to plot immediately.

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

extend_posterior_plot!(posterior_plot, f_ml, y, :red, "exact, learned params");




#
# Missing Data
#

# Create a new data set in which a chunk of the data is marked as missing in the first two
# output dimensions.
Y_missing = Matrix{Union{Missing, eltype(y.X)}}(undef, size(y.X));
Y_missing .= y.X;
Y_missing[1:2, 25:150] .= missing;
y_missing = ColVecs(Y_missing);

# # Perform inference using the true model, generate samples, compute marginal statistics.
# f_post_missing = posterior(f(x), y_missing)
# fs_post_missing = OILMMs.rand_latent(rng, f_post_missing(x_test))
# marginals_post_missing = marginals(f_post_missing(x_test))

# Plot the first couple of posterior processes.
# We'll display them later once we've done learning-related stuff below.
extend_posterior_plot!(posterior_plot, f, y_missing, :green, "approx, true params");

# true_post__missing_colour = :green
# first_marginals_missing = marginals_post_missing.X[1, :];
# plot!(posterior_plot, x_test, fs_post_missing.X[1, :];
#     linewidth=1, linealpha=0.7, color=true_post__missing_colour, label="",
# );
# plot!(posterior_plot,
#     x_test,
#     [mean.(first_marginals_missing) mean.(first_marginals_missing)];
#     linewidth=0.0,
#     color=true_post__missing_colour,
#     fillrange=[
#         quantile.(first_marginals_missing, 0.01),
#         quantile.(first_marginals_missing, 0.99),
#     ],
#     fillalpha=0.3,
#     label="",
# );
# plot!(posterior_plot, x_test, mean.(first_marginals_missing);
#     linewidth=2, linealpha=1, color=true_post__missing_colour, label="approx posterior (true params)",
# );
# # display(posterior_plot); # uncomment to show plot now.

# Regularise the nlml very slightly for stability -- best not thought of as a diffuse prior.
nlml_missing(θ::Vector) = -logpdf(build_model(θ)(x), y) + 1e-3 * sum(abs2, θ)

θ0 = randn(rng, dim_observed * dim_latent + 1 + dim_latent + dim_latent);
opts = Optim.Options(show_trace=true, iterations=300);
results = Optim.optimize(
    nlml_missing,
    θ->gradient(nlml_missing, θ)[1],
    θ0,
    BFGS(),
    opts;
    inplace=false,
);
f_ml_missing = build_model(results.minimizer);

# Visualise the results.
extend_posterior_plot!(
    posterior_plot, f_ml_missing, y_missing, :purple, "approx, learned params",
)
display(posterior_plot);

# # Perform inference using the true model, generate samples, compute marginal statistics.
# f_post_ml_missing = posterior(f_ml_missing(x), y_missing)
# fs_post_ml_missing = OILMMs.rand_latent(rng, f_post_ml_missing(x_test))
# marginals_post_ml_missing = marginals(f_post_ml_missing(x_test))

# # Plot the first couple of posterior processes.
# # We'll display them later once we've done learning-related stuff below.
# true_post__missing_colour = :green
# first_marginals_ml_missing = marginals_post_missing.X[1, :];
# plot!(posterior_plot, x_test, fs_post_missing.X[1, :];
#     linewidth=1, linealpha=0.7, color=true_post__missing_colour, label="",
# );
# plot!(posterior_plot,
#     x_test,
#     [mean.(first_marginals_missing) mean.(first_marginals_missing)];
#     linewidth=0.0,
#     color=true_post__missing_colour,
#     fillrange=[
#         quantile.(first_marginals_missing, 0.01),
#         quantile.(first_marginals_missing, 0.99),
#     ],
#     fillalpha=0.3,
#     label="",
# );
# plot!(posterior_plot, x_test, mean.(first_marginals_missing);
#     linewidth=2, linealpha=1, color=true_post__missing_colour, label="approx posterior (true params)",
# );

