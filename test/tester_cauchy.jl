# Ecrire les tests de l'algorithme du pas de Cauchy
using Test


"""
Tester le pas de Cauchy

# Les cas de test (dans l'ordre)

    - fct 1 : x⁴ + x³ cas a > 0 en -1
    - fct 2 : x cas a = 0
    - fct 3 : x⁴ + x³ - x⁵ cas a < 0 en 1.1

"""
function tester_cauchy(cauchy::Function)

    fct1(x)=x[1]^4 + x[1]^3
    # la gradient de la fonction fct1
    grad_fct1(x)=[4*x[1]^3 + 3*x[1]^2;0]
    #la hessienne de la fonction fct1
    hess_fct1(x)=[12*x[1]^2 + 6*x[1] 0;0 0]

    fct2(x) = x[1]
    # la gradient de la fonction fct2
    grad_fct2(x)= [1;0]
    #la hessienne de la fonction fct2
    hess_fct2(x)=[0 0;0 0]

    fct3(x)= -x[1]^5 + x[1]^4 + x[1]^3
    # la gradient de la fonction fct3
    grad_fct3(x)= [-5*x[1]^4 + 4*x[1]^3 + 3*x[1]^2;0]
    #la hessienne de la fonction fct3
    hess_fct3(x)= [-20*x[1]^3 + 12 * x[1]^2 + 6*x[1] 0;0 0]

	Test.@testset "Pas de Cauchy" begin
        Test.@testset "test f1, a>0" begin
            x = -1
            Test.@testset "test minimum local" begin
                Δ = 0.5
                s = cauchy(grad_fct1(x), hess_fct1(x), Δ)
                Test.@test isapprox(s[1],0.25,atol=0.1)
                Test.@test isapprox(s[2],0,atol=0.1)
            end

            Test.@testset "test descente vers minimum local" begin
                Δ = 0.1
                s = cauchy(grad_fct1(x), hess_fct1(x), Δ)
                Test.@test isapprox(s[1],1-0.921218,atol=0.03)
                Test.@test isapprox(s[2],0,atol=0.1)
            end
        end

        Test.@testset "test f2, a=0" begin
            x = 0
            Test.@testset "test minimum local" begin
                Δ = 1
                s = cauchy(grad_fct2(x), hess_fct2(x), Δ)
                Test.@test isapprox(s[1],-1,atol=0.05)
                Test.@test isapprox(s[2],0,atol=0.1)
            end
        end

        Test.@testset "test f3, a<0" begin
            x = 1.2
            Test.@testset "test pente montante" begin
                Δ = 0.05
                s = cauchy(grad_fct3(x), hess_fct3(x), Δ)
                Test.@test isapprox(s[1],-1.2+1.16516,atol=0.1)
                Test.@test isapprox(s[2],0,atol=0.1)
            end
        end

    end

end