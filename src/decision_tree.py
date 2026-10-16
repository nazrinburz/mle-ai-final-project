"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)
        self.proba=None
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """Gini / entropy (classification) or MSE/variance (regression)."""
        y=np.asarray(y)
        if y.size==0:
            return 0.0

        if self.task=="regression":
            if self.criterion!="mse":
                raise ValueError("Regression requires criterion='mse'")
            mean=np.mean(y)
            mse=np.mean((y-mean)**2)
            return float(mse)
        elif self.task=="classification":
            classes,counts=np.unique(y,return_counts=True)
            probabilities=counts/y.size
            
            if self.criterion=='gini':
                return float(1.0-np.sum(probabilities**2))
            elif self.criterion=='entropy':
                return float(-np.sum(probabilities*np.log2(probabilities)))
            else:
                raise ValueError('Unsupported criterion')

    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        n_samples,n_features=X.shape
        parent_impurity=self._impurity(y)

        best_split=None
        best_gain=0.0
        for feature in range(n_features):
            values=np.unique(X[:,feature])
            if values.size<2:
                continue
            thresholds=values[:-1]/2.0+values[1:]/2.0

            for threshold in thresholds:
                left_mask=X[:,feature]<=threshold
                right_mask= ~ left_mask
                y_left=y[left_mask]
                y_right=y[right_mask]

                n_left=y_left.size
                n_right=y_right.size

                if(n_left<self.min_samples_leaf or n_right<self.min_samples_leaf):
                    continue
                left_impurity=self._impurity(y_left)
                right_impurity = self._impurity(y_right)

                weighted_impurity=((n_left/n_samples)*left_impurity + (n_right/n_samples)*right_impurity)
                gain=parent_impurity-weighted_impurity
                if gain>best_gain:
                    best_gain=gain
                    best_split=(feature,float(threshold),float(gain))
        return best_split

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        node=Node()
        if self.task=="classification":
            counts=np.bincount(y.astype(int),minlength=2)
            node.value=int(np.argmax(counts))
            node.proba=counts/y.size
        else:
            node.value=float(np.mean(y))
        if self.max_depth is not None and depth>=self.max_depth:
            return node
        if y.size<self.min_samples_split:
            return node
        if self._impurity(y)<=0.0:
            return node
        
        split=self._best_split(X, y)

        if split is None:
            return node

        feature, threshold, gain=split

        weighted_gain=(y.size/self.n_samples_)*gain

        if weighted_gain<self.min_impurity_decrease:
            return node
        node.feature=feature
        node.threshold=threshold

        left_mask=X[:,feature]<=threshold
        right_mask= ~ left_mask
        node.left=self._grow(X[left_mask],y[left_mask],depth+1)
        node.right=self._grow(X[right_mask],y[right_mask],depth + 1
        )
        return node




    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.n_samples_ = y.size
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")
