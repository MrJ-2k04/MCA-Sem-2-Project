import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("Bike Price Model Evaluation")

# Load data
df = pd.read_csv("Cleaned_Bike_Data.csv")


# Fit Models
@st.cache_resource
def fit_models():
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
    model_wrappers = {
        "Linear Regression": {"regressor": LinearRegression()},
        "Decision Tree": {"regressor": DecisionTreeRegressor(max_depth=5, random_state=0)},
        "Random Forest": {"regressor": RandomForestRegressor(n_estimators=100, random_state=0)},
        "Gradient Boosting": {"regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)},
    }

    for model_wrapper in model_wrappers.values():
        # Train models
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", model_wrapper["regressor"])])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        model_wrapper["model"] = pipeline
        model_wrapper["r2"] = r2
        model_wrapper["mae"] = mae
        model_wrapper["rmse"] = rmse

    return model_wrappers


# Render UI
def render_ui(model_wrappers):
    # User selects model
    selected_model_name = st.selectbox("Select a model to evaluate:", list(model_wrappers.keys()))
    model_wrapper = model_wrappers[selected_model_name]

    r2 = model_wrapper["r2"]
    mae = model_wrapper["mae"]
    rmse = model_wrapper["rmse"]

    # Display metrics
    if st.button("Evaluate Model"):
        st.subheader(f"Results for {selected_model_name}")
        st.metric("Accuracy", f"{r2 * 100:.2f}%")
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

        st.write(
            "Note: R² score closer to 1 indicates better model fit. Lower MAE and RMSE indicate better prediction accuracy."
        )


# Main
model_wrappers = fit_models()
render_ui(model_wrappers)