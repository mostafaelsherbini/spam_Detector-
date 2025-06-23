
import streamlit as st
import joblib

# تحميل النموذج و الـ vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# عنوان التطبيق
st.title("📩 Spam Detector")

# إدخال المستخدم
msg = st.text_area("اكتب الرسالة النصية هنا:")

# زر التنبؤ
if st.button("تحليل"):
    vect_msg = vectorizer.transform([msg])
    prediction = model.predict(vect_msg)[0]
    st.success(f"النتيجة: **{prediction.upper()}**")
