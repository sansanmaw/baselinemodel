import streamlit as st
import torch
import re
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification

st.title("GBV Classification Tool")
st.write("Enter text to classify the type of Gender Based Violence detected.")

@st.cache_resource
def load_model():
    model_path = "gbv_final_model"
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return tokenizer, model, le

tokenizer, model, le = load_model()

user_input = st.text_area("Paste tweet or text here:")

if st.button("Analyze"):
    if user_input:
        # Simple preprocessing
        clean_text = user_input.lower().strip()
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        prediction = torch.argmax(logits, dim=1).item()
        label = le.inverse_transform([prediction])[0]
        
        st.success(f"Predicted Category: **{label}**")
    else:
        st.warning("Please enter some text.")
