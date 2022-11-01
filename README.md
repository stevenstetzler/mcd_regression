# robust_linear_regression

Python implementation of robust linear regression in the multiple regression and multivariate regression cases using the minimum covariance determinant (MCD).

Usage ([examples/example.ipynb](examples/example.ipynb)):
```python
>>> from robust_linear_regression import RobustLinearRegression
>>> rlr = RobustLinearRegression()
>>> x = data[:, 0, None] # n x d_x, must be 2D
>>> y = data[:, 1:] # n x d_y, must be 2D
>>> rlr.fit(x, y)
>>> residuals = y - rlr.predict(x)
>>> slope = rlr.beta
>>> intercept = rlr.alpha
```

Installation:
- Development
```bash
$ git clone https://github.com/stevenstetzler/robust_linear_regression.git
$ python -m pip install -e robust_linear_regression
```

References:
- Peter J Rousseeuw, Stefan Van Aelst, Katrien Van Driessen & Jose A Gulló (2004) Robust Multivariate Regression, Technometrics, 46:3, 293-305, DOI: [10.1198/004017004000000329](https://doi.org/10.1198/004017004000000329)
