import setuptools

setuptools.setup(
    name="robust_linear_regression",
    version="0.0.1",
    author="Steven Stetzler",
    author_email="steven.stetzler@gmail.com",
    description="Robust linear regression in the multiple regression and multivariate regression cases using the minimum covariance determinant",
    packages=setuptools.find_packages(where="."),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
    ]
)
