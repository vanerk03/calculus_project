from abc import ABC, abstractmethod
import numpy as np


class BaseRegressor(ABC):
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit(self, X, y) -> 'BaseRegressor':
        pass
    
    @abstractmethod
    def calc_loss(self, X, y):
        pass 

class LinearRegressor(BaseRegressor):
    def __init__(self, system_size: tuple[int, int], max_iter: int=100, lr: float=0.001, tolerance: float=0.001) -> None:
        self.w = np.random.randn(*system_size)
        self._max_iter = max_iter
        self._lr: float = lr
        self._tolerance = tolerance
        # prolly regularization (although not sure if needed)
        # prolly add decaying lr

    def calc_loss(self, X, y):
        # for now I think mse is viable, although I begin to doubt it as I'm writing this))))
        return np.sum(np.square(X @ self.predict(X).T - y))

    def _calc_grad(self, X, y):
        pass

    def _step(self, X, y):
        self.w -= self._lr * self._calc_grad(X, y)
        
    def predict(self, X):
        return self.w

    def fit(self, X, y):
        prev_loss = np.inf
        for _ in range(self._max_iter):
            self._step()
            loss = self.calc_loss(X, y)
            if loss - prev_loss < self._tolerance:
                break
            prev_loss = loss
        return self
