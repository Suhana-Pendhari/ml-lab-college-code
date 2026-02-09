# Basic Python
# 1. Writing a function to check prime numbers
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
print("Is 7 prime?", is_prime(7))

# 2. NumPy operations
import numpy as np

# Create a 1D array
array = np.array([10, 20, 30, 40, 50])
print("Original array:", array)
print("Array multiplied by 2:", array * 2)

# 3. Pandas DataFrame
import pandas as pd
data = {"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]}
df = pd.DataFrame(data)
print("DataFrame:")
print(df)

# Adding a new column
df['Salary'] = [50000, 60000, 70000]
print("Updated DataFrame:")
print(df)

# 4. Matplotlib visualization
import matplotlib.pyplot as plt
plt.bar(df['Name'], df['Salary'])
plt.xlabel('Name')
plt.ylabel('Salary')
plt.title('Salary by Name')
plt.show()
