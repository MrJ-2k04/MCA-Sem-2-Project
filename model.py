import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("Bike Price Model Evaluation")

# Load data
df = pd.read_csv("Cleaned_Bike_Data.csv")

# Features and target
X = df[["age", "power", "brand", "owner_encoded", "city", "kms_driven"]]
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["brand", "city"]),
        ("scaler", StandardScaler(), ["age", "power", "kms_driven"]),
    ],
    remainder="passthrough",
)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=0),
    #"SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0),
}

# User selects model
selected_model_name = st.selectbox("Select a model to evaluate:", list(models.keys()))

if st.button("Evaluate Model"):
    model = models[selected_model_name]
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    st.subheader(f"Results for {selected_model_name}")
    st.metric("Accuracy", f"{r2 * 100:.2f}%")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
    

    st.write("Note: R² score closer to 1 indicates better model fit. Lower MAE and RMSE indicate better prediction accuracy.")
