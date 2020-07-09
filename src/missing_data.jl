# Implementation of projection operation under missing data. Probably most of this needs a
# custom gradient implementation.
function project(
    S::Diagonal{T},
    U::AbstractMatrix{T},
    Y::ColVecs{Union{Missing, T}},
    σ²::T,
    D::Diagonal{T},
) where {T<:Real}

    # Compute patterns and assignments of data to patterns.
    patterns, rows, idxs = compute_patterns(Y.X)

    # Construct the projection matrix for each pattern.
    Us = Dict(p => U[rows[p], :] for p in patterns)

    # Project each pattern in the data.
    sqrtS = cholesky(S).U
    Yproj_blocks = Dict(p => sqrtS \ (Us[p] \ Y.X[rows[p], idxs[p]]) for p in patterns)

    # Construct the single projected data matrix, in the correct order.
    Yproj = Matrix{T}(undef, size(U, 2), length(Y))
    foreach(patterns) do p
        Yproj[:, idxs[p]] = Yproj_blocks[p]
    end

    # Compute each block of the projected noise.
    almost_ΣT_blocks = Dict(
        p => diag(inv(cholesky(Symmetric(Us[p]'Us[p] + 1e-9I)))) for p in patterns
    )

    # Assemble blocks into a single matrix, with a different scalar observation noise
    # for each observation.
    almost_ΣT = Matrix{T}(undef, size(Yproj))
    foreach(patterns) do p
        almost_ΣT[:, idxs[p]] .= almost_ΣT_blocks[p]
    end
    ΣT = σ² .* inv(S) * almost_ΣT .+ diag(D)

    return Yproj, ΣT
end

function regulariser(
    S::Diagonal{T},
    U::AbstractMatrix{T},
    σ²::T,
    y::ColVecs{Union{Missing, T}},
) where {T<:Real}

    # Compute patterns and assignments of data to patterns.
    patterns, rows, idxs = compute_patterns(y.X)

    # Construct the projection matrix for each pattern.
    Us = Dict(p => U[rows[p], :] for p in patterns)

    # Pre-compute one term.
    logdet_S = logdet(cholesky(S))

    # Compute the regularisation term for each block.
    return sum(patterns) do pattern
        Uo = Us[pattern]
        n = length(idxs[pattern])
        p, m = size(Uo)

        chol_UotUo = cholesky(Symmetric(Uo'Uo + 1e-9I))
        Yo = y.X[rows[pattern], idxs[pattern]]

        return -(n * (logdet_S + logdet(chol_UotUo) + (p - m) * log(2π * σ²)) +
            (sum(abs2, Yo) - sum(abs2, chol_UotUo.U' \ Uo'Yo)) / σ²) / 2
    end
end

# Helper function for `project` that handles various bits of non-differentiable stuff
# that can be safely @nograd-ed.
function compute_patterns(Y::AbstractMatrix{Union{Missing, T}} where {T<:Real})

    # Compute unique missing-ness patterns.
    missingness = eachcol(ismissing.(Y))
    patterns = unique(collect.(missingness))

    # For each pattern, compute the rows of `Y` that are not missing.
    available_rows = Dict(
        pattern => filter(n -> !pattern[n], 1:size(Y, 1)) for pattern in patterns
    )

    # Add the location each column of `Y` to mapping from block-to-columns.
    idxs = Dict(pattern => Int[] for pattern in patterns)
    for (n, pattern) in enumerate(missingness)
        push!(idxs[pattern], n)
    end

    return patterns, available_rows, idxs
end

Zygote.@nograd compute_patterns
