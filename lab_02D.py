import numpy as np

class RMSProp:
    """An implementation of the RMSProp optimization algorithm.

    Attributes:
        eta (float): Learning rate coefficient.
        gamma (float): Decay factor for the moving average.
        epsilon (float): Small constant to prevent division by zero.
    """

    eta: float
    gamma: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, gamma: float = 0.9, epsilon: float = 1e-8):
        """Sets up the initial hyperparameters for the optimizer."""
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon

    def optimize(self, oracle, x0: np.ndarray, *,
                 max_iter: int = 100, eps: float = 1e-5) -> np.ndarray:
        """Runs the iterative optimization process using calculated gradients.
        
        Termination occurs if the total iterations reach `max_iter` 
        or the Euclidean norm of the gradient drops below `eps`.
        """
        x = np.copy(x0).astype(np.float64)
        
        moving_avg = np.zeros_like(x, dtype=np.float64)

        for _ in range(max_iter):
            g = oracle.gradient(x)
            
            gn = np.linalg.norm(g)
            if gn < eps:
                break
            
            moving_avg = self.gamma * moving_avg + (1.0 - self.gamma) * (g ** 2)
            
            x = x - self.eta * g / np.sqrt(moving_avg + self.epsilon)

        return x