import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Basic Python: Tax Calculator Function ---
def calculate_total(price, tax_rate=0.08):
    """Calculates the final price including tax."""
    if price < 0:
        return 0.0
    total = price + (price * tax_rate)
    return round(total, 2)

# --- 2. NumPy: Applying Bulk Discounts ---
# Let's say these are the base prices for our products
base_prices = np.array([3.00, 4.50, 4.00, 5.00])
# Apply a 10% discount to all prices using NumPy broadcasting
discounted_prices = base_prices * 0.90

# --- 3. Pandas: Creating and Updating the Dataset ---
data = {
    "Item": ["Espresso", "Latte", "Cappuccino", "Mocha"],
    "BasePrice": base_prices,
    "SalePrice": discounted_prices,
    "UnitsSold": [25, 40, 30, 20]
}

df = pd.DataFrame(data)

# Calculate Revenue for each item using the custom function logic (Price * Units)
# We apply the tax calculator function to the SalePrice
df['PriceWithTax'] = df['SalePrice'].apply(calculate_total)
df['TotalRevenue'] = df['PriceWithTax'] * df['UnitsSold']

print("--- Coffee Shop Sales Report ---")
print(df)
print("\nTotal Daily Revenue: $", df['TotalRevenue'].sum())

# --- 4. Matplotlib: Visualization ---
plt.figure(figsize=(8, 5))
plt.bar(df['Item'], df['TotalRevenue'], color='peru', edgecolor='brown')

# Adding labels and styling
plt.xlabel('Menu Item', fontsize=12)
plt.ylabel('Total Revenue ($)', fontsize=12)
plt.title('Daily Revenue Breakdown', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
