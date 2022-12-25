from abc import ABC, abstractmethod
import numpy as np


class BaseRegressor(ABC):
    @abstractmethod
    def fit_predict(self, X):
        pass

class LinearRegressor(BaseRegressor):
    """
    Don't need fit and predict here separatelly
    """
    def __init__(self, system_size: tuple[int, int], max_iter: int=100, lr: float=0.003, tolerance: float=0.001) -> None:
        self.w = np.random.randn(*system_size)
        self._max_iter = max_iter
        self._lr: float = lr
        self._tolerance = tolerance
        self._l1 = 0.05
        self._l2 = 0.01
        self._batch_size = 64
        # prolly regularization (although not sure if needed)
        # prolly add decaying learning rate

    def _calc_loss(self, X, y):
        return np.sum(np.abs(X @ self.w.T - y)) + self._l1 * np.sum(np.abs(self.w)) + self._l2 * np.sum(np.square(self.w))

    def _calc_grad(self, X, y):
        h = 0.00001
        
        idx = np.random.randint(0, X.shape[0], self._batch_size)
        X, y = X[idx], y[idx]

        grad = np.zeros(self.w.shape)
        # couldn't find it by hand cuz of the hadamard square((((((
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w[i][j] -= h
                left_loss = self._calc_loss(X, y)
                self.w[i][j] += 2 * h
                right_loss = self._calc_loss(X, y)
                self.w[i][j] -= h
                grad[i][j] = (right_loss - left_loss)

        return grad / X.shape[0] / h / 2

    def _step(self, X, y):
        # print("grad:\n", self._calc_grad(X, y))
        self.w -= self._lr * self._calc_grad(X, y)
        
    def fit_predict(self, X, y):
        prev_loss = np.inf
        for _ in range(self._max_iter):
            self._step(X, y)
            # print("coef:\n", self.w)
            loss = self._calc_loss(X, y)
            print(f"loss: {loss: .4f}")
            if abs(loss - prev_loss) < self._tolerance:
                break
            prev_loss = loss
        return self.w
