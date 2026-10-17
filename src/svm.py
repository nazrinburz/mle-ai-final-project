"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
        lr_schedule: str = "pegasos", # "pegasos": 1/(lambda*t), "inv_sqrt": 1/(lambda*sqrt(t))
        n_rff: int = 0,               # > 0: Random Fourier Features (approximate RBF kernel)
        gamma: float | None = None,   # RBF width for RFF: None -> 1/ n_features
    ):
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1.")
        if lr_schedule not in ("pegasos", "inv_sqrt"):
            raise ValueError("lr_schedule must be 'pegasos' or 'inv_sqrt'.")
        if n_rff < 0:
            raise ValueError("n_rff must be >= 0.")
        
        self.lambda_ = float(lambda_)
        self.n_iters = int(n_iters)
        self.random_state = random_state
        self.lr_schedule = lr_schedule
        self.n_rff = int(n_rff)
        self.gamma = gamma
        self.w = None              # weight vector
        self.b = 0.0               # bias

        # Scaling statistics learned from trainign data only
        self.mean_ = None 
        self.scale_ = None

        # Rows of [iteration, objective] recorded during training
        self.objective_history = []

        # Boolean mask over the training rows: y_i * f(x_i) <= 1.
        self.support_mask_ = None

        self.n_features_in_ = None
        self.classes_ = np.array([0,1])

    @staticmethod
    def _to_pm1(y):
        """Converting binary labels {0, 1} to {-1, +1}."""
        y = np.asarray(y).reshape(-1)

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only labels 0 and 1.")

        return np.where(y == 1, 1.0, -1.0)

    def _standardize(self, X):
        """Applying the scaling parameters learned from training data."""
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, "
                             f"received {X.shape[1]}.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numeric values.")

        return (X - self.mean_) / self.scale_

    def _rff_map(self, X_scaled):
        """Random Fourier Features: approximate RBF kernel with a cosine map."""
        return np.sqrt(2.0 / self.n_rff) * np.cos(X_scaled @ self._rff_W + self._rff_phase)

    def _features(self, X):
        """Scaling with train statistics, then (optionally) apply the RFF map."""
        X_scaled = self._standardize(X)
        return self._rff_map(X_scaled) if self.n_rff > 0 else X_scaled

    def _objective(self, X, y_pm1):
        """
        Regularized hinge-loss objective:
        
        J(w, b) = lambda_/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b)))
        """

        scores = X @ self.w + self.b
        hinge_loss = np.maximum(0.0, 1.0 - y_pm1 * scores)

        return float((self.lambda_ / 2.0) * np.dot(self.w, self.w) + np.mean(hinge_loss))

    def fit(self, X, y):
        """
        Training the SVM using stochastic Pegasos updates.
        Returns self.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("Training data can not be empty.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite numeric values.")

        y_pm1 = self._to_pm1(y)

        if np.unique(y_pm1).size != 2:
            raise ValueError("Training data must contain both classes: 0 and 1.")

        self.n_features_in_ = X.shape[1]

        # Fit scaling using training data only
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)

        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0] = 1.0

        X_scaled = (X - self.mean_) / self.scale_

        rng = np.random.default_rng(self.random_state)

        # Non-linearity: Random Fourier Features (RBF kernel approximation)
        # Frequencies are random (no data statistics), then the Linear solver runs on the mapped features
        if self.n_rff > 0:
            gamma = self.gamma if self.gamma is not None else 1.0 / X_scaled.shape[1]
            self._rff_W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(X_scaled.shape[1], self.n_rff))
            self._rff_phase = rng.uniform(0.0, 2.0 * np.pi, size = self.n_rff)
            X_scaled = self._rff_map(X_scaled)

        n_samples, n_features = X_scaled.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        #Record around 200 objective values, plus the start and the final value
        checkpoint = max(1, self.n_iters // 200)
        history = [(0, self._objective(X_scaled, y_pm1))]

        for t in range(1, self.n_iters + 1):
            # Selecting one random training example
            i = rng.integers(n_samples)
            x_i = X_scaled[i]
            y_i = y_pm1[i]

            # Pegasos learning rate
            if self.lr_schedule == "pegasos":
                eta = 1.0 / (self.lambda_ * t)
            else:
                eta = 1.0 / (self.lambda_ * np.sqrt(t))

            # Current signed margin
            margin = y_i * (np.dot(self.w, x_i) + self.b)

            # Regularization update applies at every step
            self.w *= 1.0 - eta * self.lambda_

            # Hinge-loss sub-gradient update
            if margin < 1.0:
                self.w += eta * y_i * x_i
                # The bias is not regularized and eta is far too large for it, so it uses a smaller decaying step size
                self.b += y_i / np.sqrt(t)

            if t % checkpoint == 0 or t == self.n_iters:
                history.append((t, self._objective(X_scaled, y_pm1)))

        self.objective_history = np.asarray(history)

        # Support vectors on the training set: y_i * f(x_i) <= 1
        self.support_mask_ = self.support_vector_mask(X, y)

        return self

    def decision_function(self, X):
        """
        Returning one signed decision score per row
        Higher scores indicate the premium class
        """
        if self.w is None:
            raise RuntimeError("Call fit(X, y) before prediction.")

        features = self._features(X)
        scores = features @ self.w + self.b

        # Always return shape (n,)
        return np.asarray(scores, dtype=float).reshape(-1)

    def predict(self, X):
        """
        Returning class predictions in the external format:
        0 = standard, 1 = premium
        """

        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)

    @property
    def margin_width_(self):
        """ Margin width 2 / ||w|| (in standardized feature space) """
        if self.w is None:
            raise RuntimeError("Call fit(X, y) first")
        norm = np.linalg.norm(self.w)
        return float("inf") if norm == 0 else float(2.0 / norm)

    def support_vector_mask(self, X, y):
        """ Boolean (n,) mask of rows with y_i * f(x_i) <= 1 """
        y_pm1 = self._to_pm1(y)
        return y_pm1 * self.decision_function(X) <= 1.0

    def support_vector_indices(self):
        """ Indices of the support vectors among the training rows """
        if self.support_mask_ is None:
            raise RuntimeError("Call fit(X, y) first.")
        return np.flatnonzero(self.support_mask_)