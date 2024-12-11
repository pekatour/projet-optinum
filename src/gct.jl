using LinearAlgebra
"""
Approximation de la solution du problème 

    min qₖ(s) = s'gₖ + 1/2 s' Hₖ s, sous la contrainte ‖s‖ ≤ Δₖ

# Syntaxe

    s = gct(g, H, Δ; kwargs...)

# Entrées

    - g : (Vector{<:Real}) le vecteur gₖ
    - H : (Matrix{<:Real}) la matrice Hₖ
    - Δ : (Real) le scalaire Δₖ
    - kwargs  : les options sous formes d'arguments "keywords", c'est-à-dire des arguments nommés
        • max_iter : le nombre maximal d'iterations (optionnel, par défaut 100)
        • tol_abs  : la tolérence absolue (optionnel, par défaut 1e-10)
        • tol_rel  : la tolérence relative (optionnel, par défaut 1e-8)

# Sorties

    - s : (Vector{<:Real}) une approximation de la solution du problème

# Exemple d'appel

    g = [0; 0]
    H = [7 0 ; 0 2]
    Δ = 1
    s = gct(g, H, Δ)

"""
function gct(g::Vector{<:Real}, H::Matrix{<:Real}, Δ::Real; 
    max_iter::Integer = 100, 
    tol_abs::Real = 1e-10, 
    tol_rel::Real = 1e-8)

    
    
    function solve²(a,b,c)
        """ Uniquement si le déterminant est > 0"""
        s1 = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
        s2 = (-b - sqrt(b^2 - 4*a*c)) / (2*a)
        return s1,s2
    end

    function q_moins_f(s) return transpose(g)*s + 0.5*transpose(s)*H*s end

        
    nb_iters=0
    gk=g
    pk=-g
    sk = zeros(length(g))
    arret = false

    if norm(gk) <= max(tol_rel*norm(g),tol_abs)
        arret = true
    elseif nb_iters == max_iter
        arret = true
    end

    while !arret
        κk = transpose(pk)*H*pk
        if κk <=0
            σk1,σk2 = solve²(transpose(pk)*pk,
                            2*transpose(sk)*pk,
                            transpose(sk) * sk -Δ^2)
            if q_moins_f(σk1*pk + sk) < q_moins_f(σk2*pk + sk)
                σk=σk1
            else
                σk=σk2
            end
            return sk + σk*pk
        end

        αk = (transpose(gk)*gk)/κk
        if norm(sk + αk*pk) >= Δ
            σk,_ = solve²(transpose(pk)*pk,2*transpose(sk)*pk,transpose(sk) * sk-Δ^2)
            return sk + σk*pk
        end
        sk = sk + αk*pk
        g_p = gk
        gk = gk + αk*H*pk
        βk = (transpose(gk) * gk) / (transpose(g_p) * g_p)
        pk = -gk + βk*pk
        nb_iters+=1

        if norm(gk) <= max(tol_rel*norm(g),tol_abs)
        	arret = true
        elseif nb_iters == max_iter
        	arret = true
        end
    end

   return sk
end
