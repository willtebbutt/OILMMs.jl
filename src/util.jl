function ChainRulesCore.rrule(::typeof(eachrow), X::VecOrMat)
    eachrow_pullback(ΔΩ::Tangent) = (NoTangent(), ΔΩ.f.A)
    return eachrow(X), eachrow_pullback
end

function ChainRulesCore.rrule(::typeof(inv), D::Diagonal{<:Real})
    Ω = inv(D)
    function inv_Diagonal_pullback(ΔΩ::NamedTuple{(:diag,)})
        return (NoTangent(), (diag = .-ΔΩ.diag .* Ω.diag .^2,))
    end
    function inv_Diagonal_pullback(ΔΩ::Diagonal)
        return (NoTangent(), Diagonal(.-ΔΩ.diag .* Ω.diag .^2))
    end
    return Ω, inv_Diagonal_pullback
end

function ChainRulesCore.rrule(::typeof(inv), C::Cholesky{<:BLAS.BlasFloat, <:StridedMatrix})
    Ω = inv(C)
    function inv_Cholesky_pullback(ΔΩ::StridedMatrix{<:BLAS.BlasFloat})
        return (NoTangent(), (factors = -C.U' \ (ΔΩ + ΔΩ') * Ω, ))
    end
    function inv_Cholesky_pullback(ΔΩ::AbstractMatrix{<:BLAS.BlasFloat})
        return inv_Cholesky_pullback(collect(ΔΩ))
    end
    return Ω, inv_Cholesky_pullback
end
