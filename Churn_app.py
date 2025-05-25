import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# تحميل النموذج (أو تدريبه إذا لم يكن موجود)
try:
    with open('churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
except:
    # تدريب نموذج بسيط لاستخدامه كبديل
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # بيانات وهمية للتدريب
    np.random.seed(0)
    data = pd.DataFrame({
        'Age': np.random.randint(18, 65, 200),
        'Tenure': np.random.randint(0, 10, 200),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Churn': np.random.choice([0, 1], 200)
    })

    # ترميز الجنس
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])

    # تدريب النموذج
    X = data[['Age', 'Tenure', 'Gender']]
    y = data['Churn']
    model = LogisticRegression()
    model.fit(X, y)

    # حفظ النموذج
    with open('churn_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# واجهة Streamlit
st.title("Customer Churn Prediction")

# مدخلات المستخدم
age = st.slider("Age", 18, 70, 30)
tenure = st.slider("Tenure (years)", 0, 10, 3)
gender = st.selectbox("Gender", ['Male', 'Female'])

# تحويل الجنس لقيمة رقمية
gender_encoded = 1 if gender == 'Male' else 0

# عمل توقع
if st.button("Predict"):
    input_data = np.array([[age, tenure, gender_encoded]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"### Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.write(f"### Probability of Churn: {probability * 100:.2f}%")
