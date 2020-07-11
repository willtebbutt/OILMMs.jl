ZygoteRules.@adjoint function eachrow(X::VecOrMat)
    function eachrow_pullback(ΔΩ::NamedTuple)
        return (ΔΩ.f.A, )
    end
    return eachrow(X), eachrow_pullback
end

ZygoteRules.@adjoint function inv(D::Diagonal{<:Real})
    Ω = inv(D)
    function inv_Diagonal_pullback(ΔΩ::NamedTuple{(:diag,)})
        return ((diag = .-ΔΩ.diag .* Ω.diag .^2,), )
    end
    function inv_Diagonal_pullback(ΔΩ::Diagonal)
        return (Diagonal(.-ΔΩ.diag .* Ω.diag .^2), )
    end
    return Ω, inv_Diagonal_pullback
end

ZygoteRules.@adjoint function inv(C::Cholesky{<:BLAS.BlasFloat, <:StridedMatrix})
    Ω = inv(C)
    function inv_Cholesky_pullback(ΔΩ::StridedMatrix{<:BLAS.BlasFloat})
        return ((factors = -C.U' \ (ΔΩ + ΔΩ') * Ω, ), )
    end
    function inv_Cholesky_pullback(ΔΩ::AbstractMatrix{<:BLAS.BlasFloat})
        return inv_Cholesky_pullback(collect(ΔΩ))
    end
    return Ω, inv_Cholesky_pullback
end

Zygote.accum(A::Diagonal, B::NamedTuple{(:diag,)}) = Diagonal(A.diag + B.diag)
Zygote.accum(A::NamedTuple{(:diag,)}, B::Diagonal) = Zygote.accum(B, A)
