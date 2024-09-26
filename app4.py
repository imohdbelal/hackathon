import os
import streamlit as st
import pandas as pd
import numpy as np
# import openai
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from dotenv import load_dotenv
import statistics
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Optional
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from typing import List, Optional, Union

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
DIMENSION = int(os.getenv("DIMENSION"))
METRIC = os.getenv("METRIC")
CLOUD = os.getenv("CLOUD")
REGION = os.getenv("REGION")
GOOGLE_APPLICATION_CREDENTIALS = service_account.Credentials.from_service_account_file("rag-learn-435708-b315be6933d4.json")

vertexai.init(project=os.getenv("PROJECT"), location=os.getenv("LOCATION"), credentials=GOOGLE_APPLICATION_CREDENTIALS)

# Initialize Pinecone using the Pinecone class
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, if not, create a new one
if INDEX_NAME not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION)  # Modify the cloud and region as needed
    )

# Initialize the index
index = pinecone.Index(INDEX_NAME)

# Initialize OpenAI and LangChain
# openai.api_key = OPENAI_API_KEY

# embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
# def get_embedding(text):
#     response = openai.embeddings.create(
#         input=text,
#         model="text-embedding-ada-002"
#     )
#     return response.data[0].embedding

def get_embedding(
    texts: Union[str, List[str]],
    task: str = "RETRIEVAL_DOCUMENT",
    dimensionality: Optional[int] = 256,
) -> List[List[float]]:
    
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        return None
    
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)
    return [embedding.values for embedding in embeddings]


# llm = OpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)
# def generate_text(prompt):
#     response = openai.chat.completions.create(
#         model="gpt-4-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=1000
#     )
#     return response.choices[0].message.content

def generate_text(prompt):
    vertexai.init(project="rag-learn-435708", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat()
    output = chat.send_message([prompt], generation_config=generation_config)
    return output.candidates[0].content.parts[0].text

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Replace usage of `llm` with `generate_text`


# Create a Pinecone Vector Store with LangChain
def create_pinecone_store(index):
    # Define a lambda or function to get embeddings
    def embed_query(text):
        return get_embedding(text)
    # Pass the lambda function to PineconeVectorStore
    return PineconeVectorStore(index=index, embedding=embed_query, text_key="description", namespace="ns1")

# Initialize PineconeVectorStore with the text_key argument
vector_store = create_pinecone_store(index)


# Sidebar for file upload
st.sidebar.title("Bug Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset CSV (optional)", type=['csv'])

# Function to clean metadata
def clean_metadata(metadata):
    for record in metadata:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = ""  # Replace NaN with empty string or suitable default
    return metadata


# Function to upload embeddings to Pinecone
def upload_embeddings_to_pinecone(df):
    try:
        # Prepare the data for uploading
        texts = df['description'].tolist()
        ids = df['bug_id'].astype(str).tolist()
        embeddings = [get_embedding(text) for text in texts]
        
       # Prepare metadata (excluding 'description' column)
        metadata_list = df.to_dict(orient='records')
        metadata_list = clean_metadata(metadata_list)
        # print(metadata_list)

        # Ensure metadata and embeddings have the same length
        if len(embeddings) != len(metadata_list):
            raise ValueError("Mismatch between number of embeddings and metadata records.")
        
        # Combine IDs, embeddings, and metadata
        vectors_with_metadata = [(ids[i], embeddings[i], metadata_list[i]) for i in range(len(ids))]

        # Upload to Pinecone
        index.upsert(vectors=vectors_with_metadata, namespace='ns1')
        # print("Embeddings uploaded successfully!")
        # print(index.describe_index_stats())
        st.sidebar.write("Embeddings uploaded successfully!")
    except Exception as e:
        st.error(f"Error uploading embeddings: {str(e)}")

# Set up BigQuery client
# client = bigquery.Client(credentials=GOOGLE_APPLICATION_CREDENTIALS, project='rag-learn-435708')

# Define BigQuery table ID (project_id.dataset_id.table_id)
# table_id = "rag-learn-435708.rag_1.bugs"

    # Function to upload data to bigquery
# def upload_data_to_bigquery(df):
#     try:
#         # Configure the load job to overwrite existing table data
#         job_config = bigquery.LoadJobConfig(
#             write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,  # Overwrites table
#         )

#         # Upload the DataFrame to BigQuery
#         job = client.load_table_from_dataframe(df, table_id, job_config=job_config)

#         # Wait for the job to complete
#         job.result()

#         print(f"Data uploaded successfully to {table_id}")
#         st.sidebar.write("Data uploaded to BigQuery successfully!")
#     except Exception as e:
#         st.error(f"Error uploading data to BigQuery: {str(e)}")

# Processing the uploaded dataset
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        # print(df)
        st.sidebar.write("Dataset loaded successfully!")
        upload_embeddings_to_pinecone(df)
        # upload_data_to_bigquery(df)
    except Exception as e:
        st.sidebar.error(f"Failed to process the file: {str(e)}")
else:
    st.sidebar.write("Using existing data in vector store.")


# Right side for bug description input
st.title("Bug Resolution Predictor")
bug_description = st.text_area("Enter the new bug description:")

# Debugging statement
if not isinstance(bug_description, str):
    st.error("Bug description should be a string.")

# LangChain PromptTemplate for summarization
prompt_template = """
This is the description of a new bug for which i am looking for a 
probable solution steps: "{bug_title}". Based on the similarity search,
here is the combined solution steps for all the similar bugs found from historical data: "{solution_steps}". 
Based on the description and the combined solution steps of historical bugs, give
me a structured solution steps for the new bug.
"""

# Function to generate AI response using LangChain
def generate_summary(bug_title, solution_steps):
    prompt = prompt_template.format(bug_title=bug_title, solution_steps=solution_steps)
    summary = generate_text(prompt)
    return summary

# Button to analyze bug
if st.button("Analyze Bug"):
    if bug_description:
        try:
            # Generate embedding for new bug description
            print("Generating new embedding")
            new_bug_embedding = get_embedding(bug_description)
            print("New embedding generated")
            # # Ensure new_bug_embedding is a list of floats
            if isinstance(new_bug_embedding, np.ndarray):
                new_bug_embedding = new_bug_embedding.tolist()

            # Search for similar bugs in Pinecone
            try:
                search_results = vector_store.similarity_search_by_vector_with_score(embedding=new_bug_embedding, k=5, namespace='ns1')
                print("Got search result")
                # Print the type of search_results for debugging
                # print("Type of search_results:", type(search_results))
                # print("Contents of search_results:", search_results)

            except Exception as e:
                st.error(f"Error during similarity search: {str(e)}")

            if search_results:
                # Filter results with score greater than 0.75
                filtered_results = [(result, score) for result, score in search_results if score > 0.75]
                # print(filtered_results)
                n_bugs = len(filtered_results)

                if n_bugs > 0:
                    st.write(f"Top {n_bugs} Similar Bugs Found:")
                    
                    # Create a table for bug details
                    bug_data = []
                    all_solution_steps = []
                    # collect resolution times
                    resolution_times = []

                    for result, score in filtered_results:
                        try:
                            # print(result.page_content)
                            # Access Document attributes
                            bug_id = result.metadata.get('bug_id', "Unknown ID") if hasattr(result, 'metadata') and result.metadata.get('bug_id') is not None else "Unknown ID"
                            if bug_id != "Unknown ID":
                                bug_id = int(bug_id)
                            # description = result.metadata.get('description', "No description available") if hasattr(result, 'metadata') else "No description available"
                            description = result.page_content

                            # Collect solution steps for summary generation
                            solution_steps = result.metadata.get('solution_steps', "No solution steps available") if hasattr(result, 'metadata') else "No solution steps available"
                            all_solution_steps.append(solution_steps)

                            # Append bug details to the list
                            bug_data.append({
                                "Bug ID": bug_id,
                                "Description": description,
                                "Similarity Score (%)": f"{score * 100:.2f}"
                            })
                            
                            if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                                resolution_time = result.metadata.get('resolution_time_hrs', None)
                                if resolution_time is not None:
                                    try:
                                        resolution_time = float(resolution_time)
                                        if resolution_time > 0:
                                            resolution_times.append(resolution_time)
                                    except ValueError:
                                        continue
                                # print(resolution_times)
                            else:
                                print("No resolution time")

                        except Exception as e:
                            st.error(f"Error processing result: {str(e)}")

                    # Display bug data in a table
                    st.dataframe(bug_data, hide_index=True)

                    # get median of resolution times
                    if resolution_times:
                        median_resolution_time = statistics.median(resolution_times)
                        print(median_resolution_time)
                    else:
                        st.write("No resolution time available")

                    # Generate AI summary based on all solution steps
                    if all_solution_steps:
                        combined_solution_steps = " ".join(all_solution_steps)  # Combine all solution steps
                        summary = generate_summary(bug_description, combined_solution_steps)  # Assuming generate_summary can handle empty description
                        st.subheader("Suggested Solution Steps:")
                        st.write(summary)
                        st.write("Based on the similar bugs found, it would take approximately ",median_resolution_time," hours to fix this issue.")
                else:
                    st.write("No similar bugs found with a score greater than 0.75.")
            else:
                st.write("No similar bugs found.")

        except Exception as e:
            st.error(f"Error during bug analysis: {str(e)}")
    else:
        st.error("Please enter a bug description.")

# Sidebar to display similar bug IDs and click for details
if 'search_results' in locals() and search_results:
    st.sidebar.subheader("Similar Bugs")
    for result, _ in search_results:
        # Print the type and contents of result for debugging
        # print("Type of result in sidebar:", type(result))
        # print("Contents of result in sidebar:", result)

        # Access attributes of Document object
        try:
            bug_id = result.metadata.get('bug_id', "Unknown ID") if hasattr(result, 'metadata') else "Unknown ID"
            description = result.metadata.get('description', "No description available") if hasattr(result, 'metadata') else "No description available"
            solution_steps = result.metadata.get('solution_steps', "No solution steps available") if hasattr(result, 'metadata') else "No solution steps available"

            if st.sidebar.button(f"Bug ID: {bug_id}"):
                st.sidebar.write(f"Description: {description}")
                st.sidebar.write(f"Solution Steps: {solution_steps}")
        except Exception as e:
            st.sidebar.error(f"Error displaying result: {str(e)}")

# Add a link to Looker Studio dash
st.sidebar.markdown(
    """
    <div style='position: fixed; bottom: 10px;'>
        <a href="https://lookerstudio.google.com/u/0/reporting/a6a55855-c46f-446a-b4ba-ddf64ffd3a63/page/djqCE" target="_blank" style='color:blue; text-decoration:none;'>
            View data overview
        </a>
    </div>
    """, unsafe_allow_html=True
)
