ZygoteRules.@adjoint function eachrow(X::VecOrMat)
    function eachrow_rrule(ΔΩ::NamedTuple)
        return (ΔΩ.f.A, )
    end
    return eachrow(X), eachrow_rrule
end

ZygoteRules.@adjoint function inv(D::Diagonal{<:Real})
    function inv_Diagonal_rrule(ΔΩ::NamedTuple{(:diag,)})
        return ((diag = .-ΔΩ.diag .* Ω.diag .^2,), )
    end
    function inv_Diagonal_rrule(ΔΩ::Diagonal)
        return (Diagonal(.-ΔΩ.diag .* Ω.diag .^2), )
    end
    Ω = inv(D)
    return Ω, inv_Diagonal_rrule
end

Zygote.accum(A::Diagonal, B::NamedTuple{(:diag,)}) = Diagonal(A.diag + B.diag)

"""
    BlockMissingColVecs


"""
struct BlockMissingColVecs{T, TX<:StridedMatrix{T}} <: AbstractVector{T}
    X::TX
    patterns::Vector{Vector{Int}}
end

function block_missing(x::ColVecs{Union{Missing, T}} where {T<:Real})

    # Determine unique patterns.
    patterns = unique(collect.(eachcol(ismissing.(X))))

    # Construct data structure that knows about these patterns.
    return BlockMissingColVecs(X, patterns)
end
