from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "model" / "ridge.pkl", 'rb') as f:
        artifacts_ = pickle.load(f)
    return artifacts_


artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
ohe_columns = artifacts["ohe_columns"]



st.title("Прогноз цены машины по введенным параметрам")
st.subheader("")
st.markdown("""
Предполагается загрузка CSV файла определенного формата для получения результата. 
Для предсказания используется модель ridge, тк во время экспериментов она дала наилучший результат из всех опробованных линейных моделей и их оптимизаций. 
Ниже для иллюстрации представлена матрица корреляций между признаками
""")

with open(BASE_DIR / "model" / "corr_matrix.pkl", 'rb') as f:
    corr_matrix = pickle.load(f)

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
st.pyplot(plt.gcf())

st.subheader("Требуемый формат входного CSV:")
st.markdown("""
**CSV должен содержать только такие колонки:**

- `name` — строка  
- `year` — 4 цифры  
- `km_driven` — целое число  
- `fuel` — Diesel / Petrol / LPG  
- `seller_type` — строка  'Individual' 'Dealer' 'Trustmark Dealer'
- `transmission` — Manual / Automatic  
- `owner` —  'Individual' 'Dealer' 'Trustmark Dealer'
- `mileage` — число
- `engine` — целое число  
- `max_power` — число 
- `seats` — одно из: 4, 5, 6, 7, 8, 9, 10, 14
""")


uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Загруженный файл:")
    st.write(df.head())

    required_columns = [
        "name", "year", "km_driven", "fuel", "seller_type",
        "transmission", "owner", "mileage", "engine",
        "max_power", "seats"
    ]

    errors = []

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"Отсутствуют обязательные колонки: {missing}")

    if not errors:
        if not df["year"].astype(str).str.fullmatch(r"\d{4}").all():
            errors.append("Колонка 'year' должна состоять из 4 цифр.")

        if not pd.api.types.is_integer_dtype(df["km_driven"]):
            errors.append("Колонка 'km_driven' должна быть целым числом.")

        allowed_fuel = {"Diesel", "Petrol", "LPG"}
        if not df["fuel"].isin(allowed_fuel).all():
            errors.append(f"Колонка 'fuel' должна содержать одно из: {allowed_fuel}")

        if not df["seller_type"].isin(["Individual", "Dealer", "Trustmark Dealer"]).all():
            errors.append("Колонка 'seller_type' должна быть: Individual, Dealer, Trustmark Dealer")

        if not df["transmission"].isin(["Manual", "Automatic"]).all():
            errors.append("Колонка 'transmission' должна быть: Manual или Automatic.")

        allowed_seats = {4, 5, 6, 7, 8, 9, 10, 14}
        if not df["seats"].isin(allowed_seats).all():
            errors.append(f"Колонка 'seats' должна быть одним из: {allowed_seats}")

        if not pd.api.types.is_integer_dtype(df["engine"]):
            errors.append("Колонка 'engine' должна быть целым числом.")

        if not df["mileage"].astype(str).str.fullmatch(r"\d+(\.\d+)?").all():
            errors.append("Колонка 'mileage' должна быть числом")

        if not df["max_power"].astype(str).str.fullmatch(r"\d+(\.\d+)?").all():
            errors.append("Колонка 'max_power' должна быть числом")

    if errors:
        st.error("Ошибки валидации CSV:")
        for e in errors:
            st.write("- ", e)
    else:
        st.success("CSV корректен!")


if uploaded_file is not None:
    names = df["name"].tolist()

    df_features = df.drop(columns=["name"])

    df_ohe = pd.get_dummies(df_features, drop_first=True)

    for col in ohe_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0

    df_ohe = df_ohe[ohe_columns]

    X_scaled = scaler.transform(df_ohe)

    preds = model.predict(X_scaled)

    st.subheader("Предсказания")

    MAX_SHOW = 10

    for name, price in list(zip(names, preds))[:MAX_SHOW]:
        st.write(f"**{name}: {price:.0f}**")

    if len(preds) > MAX_SHOW:
        st.info(f"Показаны первые {MAX_SHOW} записей из {len(preds)}.")

