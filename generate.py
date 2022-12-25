import numpy as np

# x(t)' = 2x + y
# y(t)' = -5x + 7y

def generate(system, dataset_size: int=20, scope: float = 100, random_state:int =42):
    """
    system of equations is represented by the matrix of the coefficients
    
    WARNING: dataset should not be used for training as it is an answer to the given problem 
    """
    dataset = np.random.randn(dataset_size, system.shape[1]) * scope
    return dataset @ system, dataset