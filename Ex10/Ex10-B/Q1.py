import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# ---------------- Step 1: Load and Prepare Dataset ----------------
iris = datasets.load_iris()

X = iris.data[:, :2]   # sepal length & sepal width
y = iris.target

# Use only 2 classes (binary classification)
X = X[y != 2]
y = y[y != 2]

# ---------------- Step 2: Perceptron Class ----------------
class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def _unit_step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._unit_step_function(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step_function(linear_output)


# ---------------- Step 3: Train & Test ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

perceptron = Perceptron(learning_rate=0.1, max_iter=1000)
perceptron.fit(X_train, y_train)


# ---------------- Step 4: Evaluation ----------------
predictions = perceptron.predict(X_test)
accuracy = np.mean(predictions == y_test)

print("Accuracy:", accuracy * 100, "%")


# ---------------- Step 5: Decision Boundary ----------------
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')

    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title('Decision Boundary')

    plt.show()


# Plot
plot_decision_boundary(X_test, y_test, perceptron)
