# Import Libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("Bengaluru_House_Data.csv")

# -------------------------------
# Data Preprocessing
# -------------------------------

# Remove missing values
data = data.dropna()

# Function to convert total_sqft values
def convert_sqft(x):
    try:
        if '-' in str(x):
            nums = x.split('-')
            return (float(nums[0]) + float(nums[1])) / 2
        return float(x)
    except:
        return None

# Apply conversion
data['total_sqft'] = data['total_sqft'].apply(convert_sqft)

# Remove invalid rows
data = data.dropna()

# Extract BHK from size column
data['bhk'] = data['size'].str.extract('(\d+)').astype(int)

# Select required columns
data = data[['total_sqft', 'bath', 'bhk', 'price']]

# Rename columns
data.columns = ['Area', 'Bathrooms', 'Bedrooms', 'Price']

# -------------------------------
# Prepare Training Data
# -------------------------------
X = data[['Area', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# -------------------------------
# Train Linear Regression Model
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Streamlit User Interface
# -------------------------------
st.title("🏠 Bengaluru Housing Price Prediction")

st.write("Enter property details to estimate house price.")

# User Inputs
area = st.number_input(
    "Area (sq. ft)",
    min_value=500,
    max_value=10000,
    value=1000
)

bedrooms = st.selectbox(
    "Bedrooms (BHK)",
    [1, 2, 3, 4, 5]
)

bathrooms = st.selectbox(
    "Bathrooms",
    [1, 2, 3, 4]
)

# -------------------------------
# Prediction
# -------------------------------
input_data = pd.DataFrame(
    [[area, bedrooms, bathrooms]],
    columns=['Area', 'Bedrooms', 'Bathrooms']
)

prediction = model.predict(input_data)[0]

# -------------------------------
# Display Prediction
# -------------------------------
st.subheader("Predicted Price")

st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Prediction Visualization")

fig, ax = plt.subplots()

ax.bar(["Predicted Price"], [prediction])

ax.set_ylabel("Price (Lakhs)")
ax.set_title("House Price Prediction")

st.pyplot(fig)