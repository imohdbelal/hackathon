import streamlit as st
from google.cloud import aiplatform

# Initialize Streamlit app
st.title("Google Cloud Vertex AI Authentication Test")

# Initialize Vertex AI
try:
    # Replace with your project ID and location
    project_id = "rag-learn-435708"
    location = "us-central1"
    
    aiplatform.init(project=project_id, location=location)
    
    st.success("Successfully authenticated with Google Cloud Vertex AI!")
    
    # Attempt to list available models in the project
    models = aiplatform.Model.list()  # This is where we check for models
    st.subheader("Available Vertex AI Models:")
    
    if models:
        for model in models:
            st.write(f"- {model.display_name} (ID: {model.name})")
    else:
        st.write("No models available.")

except Exception as e:
    st.error(f"Authentication failed: {str(e)}")
