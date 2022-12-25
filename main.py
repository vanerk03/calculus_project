from model import LinearRegressor
from generate import generate
import numpy as np

def main():
    """
    unlike in regular regression, here as a result we have a vector for EACH entry instead of a number, that means
    we should devise our own Gradient Descent for that problem 
    """
    system = np.array([[1, 2], [3, 5]])
    X, y = generate(system, dataset_size=500, scope=12)

    model = LinearRegressor(system.shape, max_iter=300)
    print(model.fit_predict(X, y))


if __name__ == "__main__":
    main()

