import pandas as pd

# Load the dataset
file_path = 'modified_car_dataset.csv'
df = pd.read_csv(file_path)

# Define the correct column names
year_column = 'Manufacturing Year'
price_column = 'Selling Price'  # Assuming 'Selling Price' is the column name for car prices

# Function to adjust prices based on the manufacturing year
def adjust_prices(row, min_price, max_price, min_year, max_year):
    # Normalize the year to a value between 0 and 1
    normalized_year = (row[year_column] - min_year) / (max_year - min_year)
    # Calculate the adjusted price and round to nearest integer
    adjusted_price = round(min_price + normalized_year * (max_price - min_price))
    return adjusted_price

# Get the minimum and maximum year in the dataset
min_year = df[year_column].min()
max_year = df[year_column].max()

# Define the new minimum and maximum prices
min_price = 100000
max_price = 1500000

# Apply the price adjustment function to each row
df[price_column] = df.apply(adjust_prices, axis=1, args=(min_price, max_price, min_year, max_year))

# Save the modified dataset
output_path = 'dataset.csv'
df.to_csv(output_path, index=False)

print(f"Adjusted dataset saved to {output_path}")
