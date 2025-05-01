import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv("Cleaned_Bike_Data.csv")

# Update the fit_model function to store the label encoders and categories
@st.cache_resource
def fit_model():
    X = df[["age", "power", "brand", "owner_encoded", "city", "kms_driven"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    preprocessor = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["brand", "city"]),
            ("scaler", StandardScaler(), ["age", "power", "kms_driven"]),
        ],
        remainder="passthrough",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Train Model
    model.fit(X_train, y_train)

    # Store categories for encoding
    categories = {
        "brand": preprocessor.named_transformers_["onehot"].categories_[0],
        "city": preprocessor.named_transformers_["onehot"].categories_[1]
    }

    return model, categories

# Update the render_ui function to handle unseen categories
def render_ui(model, categories):
    st.title("🏍️ PriceTrack")
    st.subheader("Unlocking Bike Market Insights")
    st.write("Enter bike details below to predict its market price.")

    # User input fields
    owner_mapping = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth Owner Or More': 4
    }
    brand_input = st.selectbox("Brand", categories["brand"], index=1)
    power_input = st.number_input("Power (CC)", min_value=50, max_value=2000, value=150, step=10)
    kms_driven_input = st.number_input("KMs Driven", min_value=0, step=100, value=5000)
    age_input = st.slider("Age (Years)", 1, 30, step=1, value=5)
    city_input = st.selectbox("City", categories["city"], index=6)
    owner_input = st.selectbox("No. of owner", list(owner_mapping.keys()))
    owner_encoded = owner_mapping[owner_input]  # Convert owner input to numeric

    # Encode inputs
    input_data = pd.DataFrame({
        "city": [city_input],
        "kms_driven": [kms_driven_input],
        "age": [age_input],
        "power": [power_input],
        "brand": [brand_input],
        "owner_encoded": [owner_encoded]
    })

    # Prediction
    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f"Estimated Bike Price: ₹{int(prediction):,}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Main
model, categories = fit_model()
render_ui(model, categories)