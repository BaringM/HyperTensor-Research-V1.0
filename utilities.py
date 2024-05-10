from sklearn.metrics import accuracy_score

def calculate_accuracy(model, X, y):
    predictions = model.predict(X)
    return accuracy_score(y, predictions)
