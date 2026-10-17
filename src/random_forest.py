"""Bootstrap regression trees built with NumPy."""

import numpy as np

from .decision_tree import DecisionTree


class RandomForestRegressor:
    def __init__(
        self, n_estimators=100, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_impurity_decrease=0.0,
        max_features="sqrt", random_state=42,
    ):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.min_impurity_decrease=min_impurity_decrease
        self.max_features=max_features
        self.random_state=random_state

    def _validate_parameters(self):
        for name,minimum in (
            ("n_estimators",1), ("min_samples_split",2),
            ("min_samples_leaf",1), ("max_depth",0),
        ):
            value=getattr(self,name)
            if name=="max_depth" and value is None:
                continue
            if (not isinstance(value,(int,np.integer))
                    or isinstance(value,(bool,np.bool_)) or value<minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        value=self.min_impurity_decrease
        if (not isinstance(value,(int,float,np.integer,np.floating))
                or isinstance(value,(bool,np.bool_))
                or not np.isfinite(value) or value<0):
            raise ValueError("min_impurity_decrease must be finite and non-negative.")
        seed=self.random_state
        if seed is not None and (
            not isinstance(seed,(int,np.integer))
            or isinstance(seed,(bool,np.bool_)) or seed<0
        ):
            raise ValueError("random_state must be None or a non-negative integer.")

    def fit(self, X, y):
        self._validate_parameters()
        X=np.asarray(X,dtype=float)
        y=np.asarray(y,dtype=float)
        if X.ndim!=2 or y.ndim!=1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0]!=y.size or y.size==0 or X.shape[1]==0:
            raise ValueError("X and y must contain matching, non-empty training data.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite values.")
        DecisionTree(max_features=self.max_features)._resolve_max_features(X.shape[1])
        rng=np.random.default_rng(self.random_state)
        estimators=[]
        for _ in range(self.n_estimators):
            indexes=rng.integers(0,y.size,size=y.size)
            seed=int(rng.integers(0,2**32))
            tree=DecisionTree(
                task="regression",criterion="mse",max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features,random_state=seed,
            )
            tree.fit(X[indexes],y[indexes])
            estimators.append(tree)
        self.estimators_=estimators
        self.n_features_in_=X.shape[1]
        self.feature_importances_=np.mean(
            [tree.feature_importances_ for tree in estimators],axis=0
        )
        return self

    def predict(self, X):
        if not hasattr(self,"estimators_"):
            raise ValueError("Call fit() before predict().")
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_:
            raise ValueError("X must be 2D with the fitted number of features.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")
        return np.mean([tree.predict(X) for tree in self.estimators_],axis=0)
