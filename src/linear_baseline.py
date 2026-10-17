"""
linear_baseline.py - Problem 8 (bonus): from scratch linear baselines for Task A.

    RidgeRegression: closed form w = (X^T X + lambda I)^-1 X^T y (normal equations)
    PegasosSVR: linear SVR, sub-gradient descent on the epsilon-insensitive loss

Both follow the team's API: fit(X, y) returns self, predict(X) returns a 1D (n,) 
float arrayin the same scale as the y given to fit (if y=log(price), the
predictions are log(price): convert back with the agreed inverse, e.g. np.expm1).

Scaling: X(and for SVR, y) are standardized with training statistics inside fit and 
the same transform is applied in predict. Pass raw features.
"""

from __future__ import annotations
import numpy as np

def _check_X(X, n_features=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} features, received {X.shape[1]}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite numeric values.")
    return X

def _check_Xy(X, y):
    X = _check_X(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.shape[0] == 0:
        raise ValueError("Training data can not be empty.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite numeric values.")
    return X, y

def _fit_scaler(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0             # constant feature: avoid dividing by zero
    return mean, scale

class RidgeRegression:
    """Ridge regression solved with the normal equations (the bias is not penalized)"""

    def __init__(self, lambda_: float = 1.0):
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0.")
        self.lambda_ = float(lambda_)
        self.w = None
        self.b = 0.0
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X, y = _check_Xy(X, y)
        self.n_features_in_ = X.shape[1]
        self.mean_, self.scale_ = _fit_scaler(X)
        Z = (X - self.mean_) / self.scale_

        # Z has mean 0, so the unpenalized bias is simply mean(y); fit w on centered y.
        self.b = float(y.mean())
        yc = y - self.b

        d = Z.shape[1]
        A = Z.T @ Z + self.lambda_ * np.eye(d)        # X^T X + lambda I
        self.w = np.linalg.solve(A, Z.T @ yc)        # solve(), not an explicit inverse 
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Call fit(X, y) before prediction.")
        X = _check_X(X, self.n_features_in_)
        return (((X - self.mean_) / self.scale_) @ self.w + self.b).reshape(-1)

class PegasosSVR:
    """
    Linear support vector regression by stochastic sub-gradient descent on

        J(w, b) = (lambda/2) ||w||^2 + mean(max(0, |y - (w.x + b)| - epsilon))

    Errors smaller than epsilon (in the units of y) cost nothing: the "epsilon tube".
    """

    def __init__(self, lambda_:float =1e-3, epsilon: float = 0.1, n_iters: int = 20_000, random_state: int = 42, average: bool = True):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0.")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1.")
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)
        self.n_iters = int(n_iters)
        self.random_state = random_state
        self.average = average                   # average the iterations of the last half (less noise)

        self.w = None
        self.b = 0.0
        self.mean_ = None
        self.scale_ = None
        self.y_mean_ = 0.0
        self.y_scale_ = 1.0
        self.objective_history = []          # rows of [iteration, J] (in standardized y units)
        self.n_features_in_ = None

    def _objective(self, Z, ys, eps):
        resid = np.abs(ys - (Z @ self.w + self.b))
        return float(0.5 * self.lambda_ * np.dot(self.w, self.w) + np.mean(np.maximum(0.0, resid - eps)))

    def fit(self, X, y):
        X, y = _check_Xy(X, y)
        n, d = X.shape
        self.n_features_in_ = d

        # Train statistics only; y is standardized too so epsilon and lambda are well-behaved.
        self.mean_, self.scale_ = _fit_scaler(X)
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        Z = (X - self.mean_) / self.scale_
        ys = (y - self.y_mean_) / self.y_scale_
        eps = self.epsilon / self.y_scale_              # epsilon given in the original y units

        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d)
        self.b = 0.0
        checkpoint = max(1, self.n_iters // 200)
        history = [(0, self._objective(Z, ys, eps))]
        w_sum, b_sum, n_avg = np.zeros(d), 0.0, 0
        start_avg = self.n_iters // 2

        for t in range(1, self.n_iters + 1):
            i = rng.integers(n)
            eta = 1.0 / (self.lambda_ * t)            # Pegasos learning rate
            resid = ys[i] - (Z[i] @ self.w + self.b)

            self.w *= 1.0 - eta * self.lambda_        # regularization shrink
            if abs(resid) > eps:                      # outside the tube
                s = np.sign(resid)
                self.w += eta * s * Z[i]
                self.b += s / np.sqrt(t)              # bias: smaller decaying step

            if self.average and t > start_avg:        # running sum of late iterates
                w_sum += self.w
                b_sum += self.b
                n_avg += 1

            if t % checkpoint == 0 or t == self.n_iters:
                history.append((t, self._objective(Z, ys, eps)))

        if self.average and n_avg > 0:                # used the averaged solution
            self.w, self.b = w_sum / n_avg, b_sum / n_avg
            history.append((self.n_iters, self._objective(Z, ys, eps)))

        self.objective_history = np.asarray(history)
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Call fit(X, y) before prediction.")
        X = _check_X(X, self.n_features_in_)
        ys = ((X - self.mean_) / self.scale_) @ self.w + self.b
        return (ys * self.y_scale_ + self.y_mean_). reshape(-1)

    def outside_tube_mask(self, X, y):
        """True for rows whose error is larger than epsilon."""
        return np.abs(np.asarray(y, dtype=float).reshape(-1) - self.predict(X)) >= self.epsilon