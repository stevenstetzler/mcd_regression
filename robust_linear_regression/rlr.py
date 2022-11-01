import numpy as np
import scipy
from scipy.stats import chi
from sklearn.covariance import MinCovDet

class RobustLinearRegression():
    def __init__(self, support_fraction=None):
        self.support_fraction = support_fraction
        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.n, self.x_dim = self.X.shape
        Y_n, self.y_dim = self.Y.shape
        if self.n != Y_n:
            raise Exception("X and Y must have the name number of points")

        self.d_r_threshold = chi.ppf(0.975, df=self.y_dim)
        self.d_x_threshold = chi.ppf(0.975, df=self.x_dim)

        self.Z = np.hstack([self.X, self.Y])
        
        # scale for numerical stability of mean/covariance matrix?
        shift_Z = - self.Z.min(axis=0) / (self.Z.max(axis=0) - self.Z.min(axis=0))
        scale_Z = 1 / (self.Z.max(axis=0) - self.Z.min(axis=0))
        A = np.diag(scale_Z)
        b = shift_Z
        self.Z = self.Z @ A + b

        mcd = MinCovDet(support_fraction=self.support_fraction)
        mcd.fit(self.Z)

        self.mu = (mcd.location_ - b) @ np.linalg.inv(A)
        self.covar = np.linalg.inv(A.T) @ mcd.covariance_ @ np.linalg.inv(A)

        self.mu_x = self.mu[:self.x_dim]
        self.mu_y = self.mu[self.x_dim:]

        self.sigma_xx = self.covar[:self.x_dim, :self.x_dim]
        self.sigma_xy = self.covar[:self.x_dim, self.x_dim:]
        self.sigma_yy = self.covar[self.x_dim:, self.x_dim:]

        self.beta = scipy.linalg.pinvh(self.sigma_xx) @ self.sigma_xy
        self.sigma_e = self.sigma_yy - self.beta.T @ self.sigma_xx @ self.beta

        self.alpha = self.mu_y - self.beta.T @ self.mu_x

        y_hat = self.predict(self.X)
        self.r = self.Y - y_hat

        self.d_r = ((self.r @ scipy.linalg.pinvh(self.sigma_e) * self.r).sum(1))**0.5
        self.d_x = (((self.X - self.mu_x) @ scipy.linalg.pinvh(self.sigma_xx) * (self.X - self.mu_x)).sum(1))**0.5

        if np.any(np.isnan(self.d_r)) or np.any(np.isnan(self.d_x)):
            raise Exception("Regression produced invalid robust distances")
        else:
            self.residual_outliers = self.d_r > self.d_r_threshold
            self.good_leverage_points = self.d_x < self.d_x_threshold
            self.outliers = self.residual_outliers
    
    def predict(self, x):
        if self.beta is None or self.alpha is None:
            raise Exception("Must fit regressor to data!")
        return x @ self.beta + self.alpha
    
