from sentence_transformers import SentenceTransformer, util
import torch

from google.cloud import storage
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_ext as ste
from google.oauth2 import service_account


#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ds-research-playground-d3bb9f4b9084.json"
gcs_credentials = {
    "type": st.secrets["gcs"]["type"],
    "project_id": st.secrets["gcs"]["project_id"],
    "private_key_id": st.secrets["gcs"]["private_key_id"],
    "private_key": st.secrets["gcs"]["private_key"],
    "client_email": st.secrets["gcs"]["client_email"],
    "client_id": st.secrets["gcs"]["client_id"],
    "auth_uri": st.secrets["gcs"]["auth_uri"],
    "token_uri": st.secrets["gcs"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcs"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcs"]["client_x509_cert_url"],
    "universe_domain": st.secrets["gcs"]["universe_domain"]
}
# storage_client = storage.Client()
credentials = service_account.Credentials.from_service_account_info(gcs_credentials)
storage_client = storage.Client(credentials=credentials, project=gcs_credentials["project_id"])

bucket_name = "datasets-datascience-team"
destination_blob_name =  "datasets-datascience-team/Streamlit_apps_datasets/Driver_tagging/known_drivers.json"

st.title("Smart Keyword Suggestor")

def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

def compute_sentence_transformers_pairwise_sim(row_entries, column_entries):
    embeddings_row_entries = model.encode(row_entries, convert_to_tensor=True)
    embeddings_column_entries = model.encode(column_entries, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings_row_entries, embeddings_column_entries)
    return cosine_scores

def get_row_indexes_with_at_least_one_match_above_threshold(row_entries, column_entries, bigness_threshold):
    cosine_scores = compute_sentence_transformers_pairwise_sim(row_entries, column_entries)
    matching_row_indexes = torch.nonzero(cosine_scores > bigness_threshold, as_tuple=True)[0].unique()
    return matching_row_indexes
    

def read_known_drivers(bucket_name, destination_blob_name, client):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(destination_blob_name)
    known_drivers_string = blob.download_as_string()
    known_drivers = json.loads(known_drivers_string)
    return known_drivers

def tag_as_drivers(framework_df,topic_colname, bigness_threshold, known_drivers):
    
    topics = framework_df[topic_colname].to_list()


    unique_topics = list(set(topics))

    matching_row_indexes = get_row_indexes_with_at_least_one_match_above_threshold(unique_topics, known_drivers, bigness_threshold)

    matched_topics = [unique_topics[index] for index in matching_row_indexes]

    tagged_framework_df = framework_df.copy()
    tagged_framework_df["Status (Driver)"] = np.nan
    tagged_framework_df.loc[tagged_framework_df[topic_colname].isin(matched_topics), "Status (Driver)"] = "Driver"
    return tagged_framework_df

if "init" not in st.session_state or not st.session_state.init:
    with st.spinner("Setting everything up..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.session_state.model = model

        st.session_state.init = True

        known_drivers = read_known_drivers(bucket_name, destination_blob_name,storage_client)
        st.session_state.known_drivers = known_drivers
else:
    model = st.session_state.model
    known_drivers = st.session_state.known_drivers

def main():

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        expected_columns = ["Topic"]
        columns_are_checked = all([elt in df.columns for elt in expected_columns])
        if columns_are_checked:
            out_df = tag_as_drivers(df,"Topic", 0.7, known_drivers)

            output_csv = convert_df_to_csv(out_df)

            ste.download_button(
                label="Download results as CSV",
                data=output_csv,
                file_name="driver_tagging.csv",
                mime="text/csv",
            )
        else:
            st.error('Make sure you have these two columns spelled that same way: {}'.format(expected_columns), icon="🚨")

if __name__ == "__main__":
    main()
