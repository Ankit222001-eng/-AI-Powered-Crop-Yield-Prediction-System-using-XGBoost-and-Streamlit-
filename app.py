# app.py
# 🌾 AI-Powered Crop Yield Prediction with XGBoost + SHAP + Streamlit

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as st

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import scipy.stats as st_dist

# ==========================
# 1. Load & Merge Multiple CSVs
# ==========================
@st.cache_data
def load_and_merge():
    yield_data = pd.read_csv("yield.csv")
    rainfall = pd.read_csv("rainfall.csv")
    temp = pd.read_csv("temp.csv")
    pesticides = pd.read_csv("pesticides.csv")
    yield_df_extra = pd.read_csv("yield_df.csv")

    # 🔹 Strip spaces from column names
    for df in [yield_data, rainfall, temp, pesticides, yield_df_extra]:
        df.columns = df.columns.str.strip()

    # --- Standardize column names ---
    yield_data = yield_data.rename(columns={"Value": "hg_ha_yield"})
    rainfall = rainfall.rename(columns={"average_rain_fall_mm_per_year": "rainfall_mm"})
    temp = temp.rename(columns={"year": "Year", "country": "Area", "avg_temp": "temperature"})
    pesticides = pesticides.rename(columns={"Value": "pesticides_tonnes"})
    yield_df_extra = yield_df_extra.rename(columns={
        "avg_temp": "temperature",
        "average_rain_fall_mm_per_year": "rainfall_mm"
    })

    # --- Merge datasets ---
    df = yield_data[["Area", "Item", "Year", "hg_ha_yield"]].merge(
        rainfall[["Area", "Year", "rainfall_mm"]], on=["Area", "Year"], how="left"
    )
    df = df.merge(temp[["Area", "Year", "temperature"]], on=["Area", "Year"], how="left")
    df = df.merge(
        pesticides[["Area", "Year", "pesticides_tonnes"]], on=["Area", "Year"], how="left"
    )

    # --- Convert yield ---
    df["yield_tons_per_ha"] = df["hg_ha_yield"] / 100.0
    df.drop(columns=["hg_ha_yield"], inplace=True, errors="ignore")

    # --- Handle extra dataset (yield_df.csv) ---
    if "hg/ha_yield" in yield_df_extra.columns:
        yield_df_extra["yield_tons_per_ha"] = yield_df_extra["hg/ha_yield"] / 100.0
        yield_df_extra.drop(columns=["hg/ha_yield"], inplace=True, errors="ignore")

    # ✅ Ensure consistent column set
    keep_cols = ["Area", "Item", "Year", "rainfall_mm", "temperature", "pesticides_tonnes", "yield_tons_per_ha"]
    df = pd.concat([df[keep_cols], yield_df_extra[keep_cols]], ignore_index=True).drop_duplicates()

    return df


# ==========================
# ==========================
# 2. Preprocessing
# ==========================
df = load_and_merge()

# 🔹 Force numeric types
for col in ["rainfall_mm", "temperature", "pesticides_tonnes", "Year", "yield_tons_per_ha"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 🔹 Drop rows with NaN in important columns
df = df.dropna(subset=["yield_tons_per_ha", "rainfall_mm", "temperature", "pesticides_tonnes", "Year"])

st.write("✅ Final merged dataset preview:", df.head())

# Encode categorical features
le_crop, le_area = LabelEncoder(), LabelEncoder()
df["Crop"] = le_crop.fit_transform(df["Item"])
df["Area_enc"] = le_area.fit_transform(df["Area"])

X = df[["Area_enc", "Crop", "Year", "rainfall_mm", "temperature", "pesticides_tonnes"]]
y = df["yield_tons_per_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# 3. Train with RandomizedSearchCV
# ==========================
param_dist = {
    "n_estimators": st_dist.randint(100, 500),
    "max_depth": st_dist.randint(3, 10),
    "learning_rate": st_dist.uniform(0.01, 0.3),
    "subsample": st_dist.uniform(0.6, 0.4),
    "colsample_bytree": st_dist.uniform(0.6, 0.4),
    "min_child_weight": st_dist.randint(1, 10),
}

@st.cache_resource
def train_model():
    rs = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=10,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    rs.fit(X_train, y_train)
    return rs.best_estimator_

model = train_model()


# ==========================
# 4. Evaluate
# ==========================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.title("🌾 AI-Powered Crop Yield Prediction System")
st.subheader("📊 Model Performance")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")


# ==========================
# 5. Predict Yield (User Input)
# ==========================
st.subheader("🔮 Predict Yield")
crop_input = st.selectbox("Select Crop", le_crop.classes_)
area_input = st.selectbox("Select Area", le_area.classes_)
year = st.number_input("Year", 1960, 2100, 2020)
rainfall = st.number_input("Rainfall (mm)", 0.0, 5000.0, 1000.0)
temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
pesticides = st.number_input("Pesticides (tonnes)", 0.0, 10000.0, 500.0)

if st.button("Predict Yield"):
    crop_encoded = le_crop.transform([crop_input])[0]
    area_encoded = le_area.transform([area_input])[0]
    input_df = pd.DataFrame({
        "Area_enc": [area_encoded],
        "Crop": [crop_encoded],
        "Year": [year],
        "rainfall_mm": [rainfall],
        "temperature": [temperature],
        "pesticides_tonnes": [pesticides]
    })
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Yield: {pred:.2f} tons/hectare")


# ==========================
# 6. SHAP Explainability

st.subheader("📌 Feature Importance (XGBoost)")

importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

st.bar_chart(importance_df.set_index("feature"))

#py -3.11 -m venv .venv
#./.venv/Scripts/python.exe --version


#./.venv/Scripts/python.exe -m pip install --upgrade pip
#./.venv/Scripts/python.exe -m pip install -r requirements.txt

#./.venv/Scripts/python.exe -m streamlit run app.py



