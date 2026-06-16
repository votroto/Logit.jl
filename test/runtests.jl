using Logit
using Test

@testset "Logit" begin
    
    @test begin
        A = [1.0 0.0; 3.0 2.0]
        B = [2.0 3.0; 0.0 1.0]
        
        result = solve_qre((A, B), 1e-4)
        
        # Check result structure
        @test result.lambda >= 0
        @test length(result.strategies) == 2
        @test length(result.strategies[1]) == 2
        @test length(result.strategies[2]) == 2
        
        # Check probabilities sum to 1
        @test isapprox(sum(result.strategies[1]), 1.0, atol=1e-6)
        @test isapprox(sum(result.strategies[2]), 1.0, atol=1e-6)
        
        # Check regret is low
        @test result.regret < 1e-3
        
        true
    end
    
end
