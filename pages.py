import streamlit as st

def page1():
    st.title("Bug Dataset")
    uploaded_file = st.file_uploader("Upload your dataset CSV (optional)", type=['csv'])
    return uploaded_file

def page2():
    st.title("Bug Resolution Predictor")
    bug_description = st.text_area("Enter the new bug description:")
    analayze_button = st.button("Analyze")
    return bug_description, analayze_button

def page3():
    st.title("Welcome to Bug resolution predictor app")
    st.subheader("Helps in your bug resoultion journey")

def page4():
    st.title("Log a new bug")
    new_bug_descricption = st.text_area("Enter the new bug description:")
    submit_button = st.button("Submit")
    return new_bug_descricption, submit_button

def navigate():
    pg = st.navigation([
        st.Page(page3, title="Welcome Page", icon=":material/home:"),
        st.Page(page1, title="Admin", icon=":material/shield_person:"),
        st.Page(page2, title="Developer", icon=":material/person:"),
        st.Page(page4, title="Issuer", icon=":material/bug_report:"),
    ])
    return pg