using InteractionNetworkModels
using Test

@testset "InteractionNetworkModels.jl" begin
    # Write your tests here.
    @testset "Multinomial term" begin
        @test log_multinomial_ratio([1,1,1,1],[1]) ≈ 0.0
        @test log_multinomial_ratio([1,2],[1,1,2]) ≈ log(2/3)
    end
end
