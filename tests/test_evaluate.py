import unittest
import numpy as np
from src import evaluate as e


class MetricTests(unittest.TestCase):
    def test_known_regression(self):
        self.assertAlmostEqual(e.rmse([10,20,30],[12,18,33]), np.sqrt(17/3))
        self.assertAlmostEqual(e.mae([10,20,30],[12,18,33]), 7/3)
        self.assertAlmostEqual(e.r2([10,20,30],[12,18,33]), .915)
        self.assertEqual(e.r2([2,2],[2,2]), 1)
        self.assertEqual(e.r2([2,2],[1,2]), 0)
        self.assertTrue(np.isnan(e.r2([1],[1])))

    def test_known_classification(self):
        np.testing.assert_array_equal(e.confusion_matrix([0,0,1,1],[0,1,0,1]), [[1,1],[1,1]])
        self.assertEqual(e.precision_recall_f1([0,0,1,1],[0,1,0,1]), (.5,.5,.5))
        self.assertEqual(e.precision_recall_f1([0,1],[0,0]), (0,0,0))

    def test_auc_order_and_ties(self):
        self.assertEqual(e.roc_auc([0,1],[0,1]), 1)
        self.assertEqual(e.roc_auc([0,1],[1,0]), 0)
        self.assertEqual(e.roc_auc([0,1],[.5,.5]), .5)
        self.assertEqual(e.roc_auc([0,0,1,1],[.1,.4,.35,.8]), .75)
        self.assertTrue(np.isnan(e.roc_auc([1,1],[.2,.3])))
        rng = np.random.default_rng(42)
        for _ in range(30):
            y = np.array([0]*10+[1]*10)
            s = rng.integers(-3,4,20)
            pair_differences = s[y==1, None] - s[y==0]
            expected = np.mean((pair_differences>0) + .5*(pair_differences==0))
            self.assertAlmostEqual(e.roc_auc(y,s), expected)

    def test_invalid_inputs(self):
        for a,b in [([],[]),([1],[1,2]),([[1]],[[1]]),([np.nan],[1]),([1],[np.inf])]:
            with self.assertRaises(ValueError):
                e.rmse(a,b)
        with self.assertRaises(ValueError):
            e.confusion_matrix([0,1],[-1,1])
        with self.assertRaises(ValueError):
            e.roc_auc([0,2],[.1,.9])

    def test_sklearn_agreement(self):
        try:
            from sklearn import metrics
        except ImportError:
            self.skipTest("sklearn unavailable; hand-calculated tests still run")
        rng = np.random.default_rng(8)
        y = rng.normal(size=100)
        p = rng.normal(size=100)
        self.assertAlmostEqual(e.rmse(y,p), np.sqrt(metrics.mean_squared_error(y,p)))
        self.assertAlmostEqual(e.mae(y,p), metrics.mean_absolute_error(y,p))
        self.assertAlmostEqual(e.r2(y,p), metrics.r2_score(y,p))
        y = rng.integers(0,2,100)
        p = rng.integers(0,2,100)
        s = rng.integers(-4,5,100)
        np.testing.assert_array_equal(e.confusion_matrix(y,p), metrics.confusion_matrix(y,p))
        expected = metrics.precision_recall_fscore_support(y,p,average="binary",zero_division=0)[:3]
        np.testing.assert_allclose(e.precision_recall_f1(y,p), expected)
        self.assertAlmostEqual(e.roc_auc(y,s), metrics.roc_auc_score(y,s))


if __name__ == "__main__":
    unittest.main()
