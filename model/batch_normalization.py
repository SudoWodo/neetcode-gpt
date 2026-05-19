import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        # 1. Convert lists to NumPy arrays for element-wise math
        X = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean)
        running_var = np.array(running_var)

        if training:
            # 2. Calculate batch statistics
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # 3. Normalize current batch
            x_hat = (X - batch_mean) / np.sqrt(batch_var + eps)

            # 4. Update running statistics using momentum
            running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            running_var = (1 - momentum) * running_var + momentum * batch_var
        else:
            # 5. Normalize using previously calculated running stats
            x_hat = (X - running_mean) / np.sqrt(running_var + eps)

        # 6. Apply scale (gamma) and shift (beta)
        y = gamma * x_hat + beta

        # 7. Round to 4 decimals and return as lists
        return (
            np.round(y, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )
