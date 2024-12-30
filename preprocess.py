import pandas as pd
import re

# Load the dataset
file_path = 'Food_Dataset.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Function to extract numeric values and unit from serving descriptions
def parse_serving(serving):
    if pd.isna(serving):  # Handle missing values
        return None, None
    match = re.search(r'([\d.]+)\s*(oz\.|sticks|cups|pieces|g|grams|lychee|lychees)', serving.lower())
    if match:
        value, unit = float(match.group(1)), match.group(2)
        if unit == 'oz.':
            value *= 28.3495  # Convert ounces to grams
        elif unit in ['g', 'grams']:
            pass  # Already in grams
        # Handle other units if necessary
        return value, unit
    return None, None

# Apply the parsing function and create new columns
data['serving_value'], data['serving_unit'] = zip(*data['serving'].map(parse_serving))

# Calculate calories per gram for rows with valid data
valid_data = data.dropna(subset=['serving_value', 'grams', 'calories'])
data['calories_per_gram'] = valid_data['calories'] / valid_data['grams']

# Use average calories per gram to fill missing values
avg_calories_per_gram = data['calories_per_gram'].mean()
data['serving_grams'] = data.apply(
    lambda row: row['calories'] / avg_calories_per_gram if pd.isna(row['serving_value']) else row['serving_value'], axis=1
)

# Drop intermediate columns if not needed
final_data = data[['food', 'grams', 'serving_grams', 'calories']]

# Save the transformed dataset
final_data.to_csv('afterPreData.csv', index=False)  # Update with desired save location

print("Final dataset saved successfully with missing values handled!")
