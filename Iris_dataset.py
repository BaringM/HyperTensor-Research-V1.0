from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Create a DataFrame to display the data in a table format
iris_df = pd.DataFrame(X, columns=data.feature_names)
iris_df['species'] = [data.target_names[i] for i in y]

# Display the first few rows of the dataframe
print(iris_df.head())
