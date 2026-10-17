import numpy as np
from src.svm import PegasosSVM

def make(n=1500, d=6, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)) * np.arange(1, d + 1) * 10 + 500 # badly scaled on purpose
    s = (X[:, 0] - 500) / 10 + 0.5 * (X[:, 1] - 500) / 20 + r.normal(scale=0.2, size = n)
    return X, (s > 0).astype(int)

def test_all():
    X, y = make()
    Xtr, ytr, Xva, yva = X[:1000], y[:1000], X[1000:1250], y[1000:1250]

    # local scaling with train statistics (only for the sklearn comparison below)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xva_s = (Xtr - mu) / sd, (Xva - mu) / sd

    m = PegasosSVM(lambda_=1e-3, n_iters=20000).fit(Xtr, ytr)
    p, s = m.predict(Xva), m.decision_function(Xva)
    assert p.shape == s.shape == (250,) and set(np.unique(p)) <= {0, 1}
    assert np.array_equal(p, (s >= 0).astype(int))

    perm = np.random.default_rng(1).permutation(250)     # row order untouched
    assert np.allclose(m.decision_function(Xva[perm]), s[perm])

    acc = (p == yva).mean()
    h = m.objective_history
    print("val acc", acc, "| J", h[0, 1], "->", h[-1, 1],
          "| margin", m.margin_width_, "| #SV", len(m.support_vector_indices()))
    assert acc > 0.9 and h[-1, 1] < h[0, 1]

    from sklearn.svm import LinearSVC.                # baseline check
    sk = LinearSVC(C=1 / (1000 * 1e-3), loss="hinge", max_iter=100000).fit(Xtr_s, ytr)
    print("sklearn acc", (sk.predict(Xva_s) == yva).mean())

def test_nonlinear_rff():
    # circular boundary: a linear SVM can not solve it, RFF can 
    r = np.random.default_rng(3)
    X = r.normal(size=(1200, 2))
    y = ((X ** 2).sum(1) > 1.4).astype(int)
    lin = PegasosSVM(1e-3, 20000).fit(X[:900], y[:900])
    rff = PegasosSVM(1e-3, 20000, n_rff=300, gamma = 1.0).fit(X[:900],y[:900])
    acc_lin = (lin.predict(X[900:]) == y[900:]).mean()
    acc_rff = (rff.predict(X[900:]) == y[900:]).mean()
    print("circle: linear acc", acc_lin, "| RFF acc", acc_rff)
    assert acc_rff > 0.9 and acc_rff > acc_lin
    assert rff.predict(X[900:]).shape == rff.decision_function(X[900:]).shape == (300,)

if __name__ == "__main__":
    test_all()
    test_nonlinear_rff()
    print("Okay")