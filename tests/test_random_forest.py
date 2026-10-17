import numpy as np
import pytest

from src.decision_tree import DecisionTree
from src.random_forest import RandomForestRegressor


def training_data():
    X=np.arange(48,dtype=float).reshape(16,3)
    y=X[:,0]*2+5
    return X,y


def test_prediction_and_importance_averaging():
    X,y=training_data()
    forest=RandomForestRegressor(n_estimators=5,max_depth=2)
    assert forest.fit(X,y) is forest
    predictions=forest.predict(X[:4])
    assert predictions.shape==(4,)
    assert len(forest.estimators_)==5
    np.testing.assert_allclose(predictions,np.mean(
        [tree.predict(X[:4]) for tree in forest.estimators_],axis=0
    ))
    np.testing.assert_allclose(forest.feature_importances_,np.mean(
        [tree.feature_importances_ for tree in forest.estimators_],axis=0
    ))
    assert forest.predict(X[:0]).shape==(0,)


def test_reproducibility():
    X,y=training_data()
    first=RandomForestRegressor(n_estimators=5,random_state=17).fit(X,y)
    second=RandomForestRegressor(n_estimators=5,random_state=17).fit(X,y)
    for left,right in zip(first.estimators_,second.estimators_):
        assert left.random_state==right.random_state
        np.testing.assert_array_equal(left.predict(X),right.predict(X))
    assert len({tree.random_state for tree in first.estimators_})==5
    expected=first.predict(X)
    first.fit(X,y)
    np.testing.assert_array_equal(first.predict(X),expected)


def test_bootstrap_preserves_pairs(monkeypatch):
    X,y=training_data()
    samples=[]
    original=DecisionTree.fit

    def capture(self, sampled_X, sampled_y):
        samples.append((sampled_X.copy(),sampled_y.copy()))
        return original(self,sampled_X,sampled_y)

    monkeypatch.setattr(DecisionTree,"fit",capture)
    RandomForestRegressor(n_estimators=3,random_state=42).fit(X,y)
    assert len(samples)==3
    has_duplicates=False
    for sampled_X,sampled_y in samples:
        assert sampled_X.shape==X.shape
        assert sampled_y.shape==y.shape
        for row,target in zip(sampled_X,sampled_y):
            matches=np.all(X==row,axis=1)
            assert matches.any()
            np.testing.assert_array_equal(y[matches],[target])
        has_duplicates|=np.unique(sampled_X,axis=0).shape[0]<len(y)
    assert has_duplicates


@pytest.mark.parametrize("max_features,count", [(None,4),(1,1),("sqrt",2)])
def test_feature_subsampling_and_refit(max_features, count):
    X=np.tile(np.arange(16)[:,None],(1,4))
    y=np.arange(16,dtype=float)
    tree=DecisionTree(task="regression",criterion="mse",max_depth=2,
                      max_features=max_features,random_state=42).fit(X,y)
    assert tree.max_features_==count
    if max_features is None:
        assert tree.root.feature==0
    else:
        # Identical columns tie, so considering all features always chooses 0.
        root_features={
            DecisionTree(task="regression",criterion="mse",max_depth=1,
                         max_features=max_features,random_state=seed)
            .fit(X,y).root.feature
            for seed in range(10)
        }
        assert len(root_features)>1
    prediction=tree.predict(X)
    root_feature=tree.root.feature
    tree.fit(X,y)
    assert tree.root.feature==root_feature
    np.testing.assert_array_equal(tree.predict(X),prediction)


@pytest.mark.parametrize("value", [0,-1,5,1.5,True,"invalid"])
def test_invalid_max_features(value):
    X,y=training_data()
    with pytest.raises(ValueError,match="max_features"):
        DecisionTree(max_features=value).fit(X,y>20)
    with pytest.raises(ValueError,match="max_features"):
        RandomForestRegressor(n_estimators=1,max_features=value).fit(X,y)


@pytest.mark.parametrize("parameters", [
    {"n_estimators":0}, {"n_estimators":True}, {"n_estimators":1.5},
    {"max_depth":-1}, {"min_samples_split":1}, {"min_samples_leaf":0},
    {"min_impurity_decrease":-1}, {"min_impurity_decrease":np.nan},
    {"random_state":-1},
])
def test_invalid_parameters(parameters):
    X,y=training_data()
    with pytest.raises(ValueError):
        RandomForestRegressor(**parameters).fit(X,y)


@pytest.mark.parametrize("X,y", [
    ([1,2],[1,2]), ([[1],[2]],[[1],[2]]), ([[1],[2]],[1]),
    (np.empty((0,2)),[]), (np.empty((2,0)),[1,2]),
    ([[np.nan]],[1]), ([[1]],[np.inf]),
])
def test_invalid_training_data(X, y):
    with pytest.raises(ValueError):
        RandomForestRegressor(n_estimators=1).fit(X,y)


def test_prediction_validation():
    forest=RandomForestRegressor(n_estimators=1)
    with pytest.raises(ValueError,match="fit"):
        forest.predict([[1]])
    X,y=training_data()
    forest.fit(X,y)
    for invalid in ([1,2,3],[[1,2]],[[1,np.inf,3]]):
        with pytest.raises(ValueError):
            forest.predict(invalid)
