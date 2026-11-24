import time

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def tune_svm_hyperparameters(X_train, y_train, X_val, y_val, param_grid):
    best_acc = -1.0
    best_params = {}

    for c in param_grid["C"]:
        for gamma in param_grid["gamma"]:
            clf = SVC(kernel="rbf", C=c, gamma=gamma)
            clf.fit(X_train, y_train)

            preds = clf.predict(X_val)
            acc = accuracy_score(y_val, preds)

            if acc > best_acc:
                best_acc = acc
                best_params = {"C": c, "gamma": gamma}

    print(f"Best val accuracy: {best_acc:.4f} (C={best_params['C']}, gamma={best_params['gamma']})")
    return best_params


def train_svm(X_train, y_train, best_params):
    model = SVC(kernel="rbf", C=best_params["C"], gamma=best_params["gamma"])

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    return model, train_time