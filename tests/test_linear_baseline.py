import numpy as np
from src.linear_baseline import RidgeRegression, PegasosSVR

def rmse(y, p): return float(np.sqrt(np.mean((y-p) ** 2)))
def r2(y, p): return float(1 - np.sum((y-p) ** 2) / np.sum((y - y.mean()) ** 2))

def make(n=1500, d=5, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)) * np.arange(1, d + 1) * 10 + 300      # badly scaled on purpose
    y = 3.0 * (X[:, 0] - 300) / 10 - 2.0 * (X[:, 1] - 300) / 20 + 5 + r.normal(scale = 0.5, size=n)
    return X, y


def test_all():
    X, y = make()
    Xtr, ytr, Xva, yva = X[:1000], y[:1000], X[1000:], y[1000:]

    # Ridge must match scikit-learn's Ridge exactly (same objective) on scaled data
    ridge = RidgeRegression(lambda_=2.0).fit(Xtr, ytr)
    p = ridge.predict(Xva)
    assert p.shape == (500,)
    from sklearn.linear_model import Ridge
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sk = Ridge(alpha = 2.0).fit((Xtr - mu) / sd, ytr)
    assert np.allclose(ridge.w, sk.coef_) and np.isclose(ridge.b, sk.intercept_)
    assert np.allclose(p, sk.predict((Xva - mu) / sd))
    perm = np.random.default_rng(1).permutation(500)          # row order untouched
    assert np.allclose(ridge.predict(Xva[perm]), p[perm])
    print("Ridge val RMSE", rmse(yva, p), "R2", r2(yva, p), "| matches sklearn")

    svr = PegasosSVR(lambda_=1e-3, epsilon=0.1, n_iters=20000).fit(Xtr, ytr)
    q = svr.predict(Xva)
    assert q.shape == (500,)
    h = svr.objective_history
    print("SVR val RMSE", rmse(yva, q), "R2", r2(yva, q), "| J", h[0, 1], "->", h[-1, 1])
    assert h[-1, 1] < h[0, 1] and r2(yva, q) > 0.95

    from sklearn.svm import LinearSVR
    sks = LinearSVR(epsilon=0.1, C = 1 / (1000 * 1e-3), loss="epsilon_insensitive", max_iter=100000, random_state=0).fit((Xtr-mu) / sd, ytr)
    print("sklearn LinearSVR val RMSE", rmse(yva, sks.predict((Xva - mu) / sd)))

if __name__ == "__main__":
    test_all()
    print("Okay")