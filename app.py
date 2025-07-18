import streamlit as st

st.title("🔘 Demo Form with Submit Button")

with st.form("my_form"):
    name = st.text_input("Enter your name")
    age = st.slider("Your age", 0, 100, 25)

    # ✅ This line must be inside the form block
    submit = st.form_submit_button("Submit")

# ✅ This part runs after clicking the button
if submit:
    st.success(f"Hello {name}, you are {age} years old!")
