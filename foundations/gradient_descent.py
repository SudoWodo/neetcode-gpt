class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        def derivation(x):
            return 2*x

        for _ in range(iterations):
            init = init - learning_rate * derivation(init)
        
        return round(init, 5)
