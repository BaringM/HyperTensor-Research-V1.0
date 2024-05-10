import numpy as np
from model import initialize_model
from data_management import load_data
from utilities import calculate_accuracy

def active_learning_loop(X_initial, y_initial, X_pool, y_pool, iterations=10):
    model = initialize_model()
    model.fit(X_initial, y_initial)
    performance_history = [calculate_accuracy(model, X_initial, y_initial)]

    for i in range(iterations):
        probas = model.predict_proba(X_pool)
        uncertainties = np.max(probas, axis=1)
        query_idx = np.argmin(uncertainties)

        X_initial = np.vstack([X_initial, X_pool[query_idx]])
        y_initial = np.append(y_initial, y_pool[query_idx])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)

        model.fit(X_initial, y_initial)
        performance_history.append(calculate_accuracy(model, X_initial, y_initial))

    return performance_history

if __name__ == "__main__":
    X_initial, X_pool, y_initial, y_pool = load_data()
    performance_history = active_learning_loop(X_initial, y_initial, X_pool, y_pool)
    print("Performance history:", performance_history)
