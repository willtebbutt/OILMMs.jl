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
    patterns, rows, idxs, perm = compute_patterns(Y.X)

    # Construct the projection matrix for each pattern.
    Us = map(row -> U[row, :], rows)

    # Project each pattern in the data.
    sqrtS = cholesky(S).U
    Yproj_blocks = map((U, r, idx) -> sqrtS \ (U \ Y.X[r, idx]), Us, rows, idxs)

    # Construct the single projected data matrix, in the correct order.
    Yproj = hcat(Yproj_blocks...)[:, perm]

    # lens = map(length, idxs)
    lens = map_length(idxs)
    almost_ΣT_blocks = map(
        (U, len) -> repeat(diag(inv(cholesky(Symmetric(U'U + 1e-9I)))), 1, len), Us, lens,
    )

    # Assemble blocks into a single matrix, with a different scalar observation noise
    # for each observation.
    almost_ΣT = hcat(almost_ΣT_blocks...)[:, perm]
    ΣT = σ² .* diag(inv(S)) .* almost_ΣT .+ diag(D)

    return Yproj, ΣT
end

function regulariser(
    S::Diagonal{T},
    U::AbstractMatrix{T},
    σ²::T,
    y::ColVecs{Union{Missing, T}},
) where {T<:Real}

    # Compute patterns and assignments of data to patterns.
    patterns, rows, idxs, perm = compute_patterns(y.X)

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
    available_rows = [filter(n -> !p[n], 1:size(Y, 1)) for p in patterns]

    # Add the location each column of `Y` to mapping from block-to-columns.
    # idxs = Dict(pattern => Int[] for pattern in patterns)
    idxs = [Int[] for pattern in patterns]
    for (n, data_pattern) in enumerate(missingness)
        for (m, pattern) in enumerate(patterns)
            if data_pattern == pattern
                push!(idxs[m], n)
            end
        end
    end

    # Compute the permutation of the data required to restore the original order.
    perm = sortperm(vcat(idxs...))

    return patterns, available_rows, idxs, perm
end

Zygote.@nograd compute_patterns

map_length(x) = map(length, x)

Zygote.@nograd map_length
