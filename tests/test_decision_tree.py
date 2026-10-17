import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.decision_tree import DecisionTree


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_classification_matches_sklearn(criterion):
    X = np.array([[2, 50], [2, 70], [2, 90], [2, 110]])
    y = np.array([0, 0, 1, 1])
    X_new = np.array([[2, 60], [2, 100]])

    tree = DecisionTree(criterion=criterion, max_depth=1).fit(X, y)
    reference = DecisionTreeClassifier(
        criterion=criterion, max_depth=1, random_state=42
    ).fit(X, y)

    assert tree.root.feature == reference.tree_.feature[0] == 1
    assert tree.root.threshold == reference.tree_.threshold[0] == 80
    np.testing.assert_array_equal(tree.predict(X_new), reference.predict(X_new))
    np.testing.assert_allclose(
        tree.predict_proba(X_new), reference.predict_proba(X_new)
    )


def test_regression_matches_sklearn():
    X = np.array([[2, 50], [2, 70], [2, 90], [2, 110]])
    y = np.array([100, 120, 200, 220])
    X_new = np.array([[2, 60], [2, 100]])

    tree = DecisionTree(task="regression", criterion="mse", max_depth=1).fit(X, y)
    reference = DecisionTreeRegressor(
        criterion="squared_error", max_depth=1, random_state=42
    ).fit(X, y)

    assert tree.root.feature == reference.tree_.feature[0] == 1
    assert tree.root.threshold == reference.tree_.threshold[0] == 80
    np.testing.assert_allclose(tree.predict(X_new), reference.predict(X_new))
