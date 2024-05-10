print("Loading data module...")

def load_data(test_size=0.95):
    print("Executing load_data function...")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    data = load_iris()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=test_size, random_state=42)

