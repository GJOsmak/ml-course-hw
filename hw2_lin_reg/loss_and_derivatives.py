import numpy as np

class LossAndDerivatives:
    def __init__(self):
        pass
    @staticmethod
    def mse_derivative(X_ref, y_ref, w_hat):
        return (2 * X_ref.T @ (X_ref @ w_hat - y_ref))/np.prod(y_ref.shape)
    
    @staticmethod    
    def mae_derivative(X_ref, y_ref, w_hat):
        return (X_ref.T @ np.sign((X_ref @ w_hat - y_ref)))/np.prod(y_ref.shape)
    @staticmethod
    def l2_reg_derivative(w_hat):
        return 2*w_hat
    
    @staticmethod
    def l1_reg_derivative(w_hat):
        return np.sign(w_hat)
    
    @staticmethod
    def mse(X_ref, y_ref, w_hat):
        return np.mean((X_ref @ w_hat - y_ref)**2)
    
    @staticmethod
    def mae(X_ref, y_ref, w_hat):
        return np.mean(np.abs(X_ref @ w_hat - y_ref))
    
    @staticmethod
    def l2_reg(w_hat):
        return np.sum(w_hat**2)
    
    @staticmethod
    def l1_reg(w_hat):
        return np.sum(np.abs(w_hat))
    
    @staticmethod
    def no_reg_derivative(w):
        return np.zeros_like(w)
    
    @staticmethod
    def no_reg(w):
        return 0.
