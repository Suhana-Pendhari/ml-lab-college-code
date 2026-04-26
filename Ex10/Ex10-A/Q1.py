import numpy as np

class McCullochPittsNeuron:
    def __init__(self, threshold):
        self.threshold = threshold

    def activation(self, x):
        return 1 if x >= self.threshold else 0

    def predict(self, inputs, weights):
        weighted_sum = np.sum(inputs * weights)
        return self.activation(weighted_sum)


# ---------------- AND Gate ----------------
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([0, 0, 0, 1])

neuron_and = McCullochPittsNeuron(threshold=2)
weights_and = np.array([1, 1])

pred_and = np.array([neuron_and.predict(x, weights_and) for x in X_and])

print("AND Gate Predictions:", pred_and)
print("AND Gate Accuracy:", np.mean(pred_and == y_and))


# ---------------- OR Gate ----------------
X_or = X_and
y_or = np.array([0, 1, 1, 1])

neuron_or = McCullochPittsNeuron(threshold=1)
weights_or = np.array([1, 1])

pred_or = np.array([neuron_or.predict(x, weights_or) for x in X_or])

print("\nOR Gate Predictions:", pred_or)
print("OR Gate Accuracy:", np.mean(pred_or == y_or))


# ---------------- NOT Gate ----------------
X_not = np.array([
    [0],
    [1]
])
y_not = np.array([1, 0])

neuron_not = McCullochPittsNeuron(threshold=0)
weights_not = np.array([-1])

pred_not = np.array([neuron_not.predict(x, weights_not) for x in X_not])

print("\nNOT Gate Predictions:", pred_not)
print("NOT Gate Accuracy:", np.mean(pred_not == y_not))