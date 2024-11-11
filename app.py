import pandas as pd
import numpy as np
import requests
import streamlit as st
from openai import AzureOpenAI
import json, os
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt
import weaviate
import weaviate.classes.config as wc
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter
from google.cloud import storage
import requests, json


load_dotenv()


# model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-6-v3')
azure_client = AzureOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],  
    api_version="2023-07-01-preview",
    azure_endpoint = "https://textgen.openai.azure.com/"
)



# Set these environment variables
URL = os.getenv("WEAVIATE_URL")
APIKEY = os.getenv("WEAVIATE_APIKEY")
  
# Connect to a WCS instance
client = weaviate.connect_to_wcs(
    cluster_url=URL,
    auth_credentials=weaviate.auth.AuthApiKey(APIKEY),
    headers={
        "X-HuggingFace-Api-Key": os.getenv("HUGGING_FACE_API")
    }
)


ms_marco_minilm = client.collections.get("Deep_search_Damac")

@retry(wait=wait_fixed(0.35), stop=stop_after_attempt(30))
def get_content_url(m_id, a_id):
    print('getting the url')
    try:
        fir_r=requests.get(
            "https://damacacademy.damacgroup.com/api/learntron/generateApiKey/DamacGroup/LearnTron2015"
        )
        sec_r=requests.get(
            "https://damacacademy.damacgroup.com/api/learntron/generateApiToken", 
            headers={
                "Learntron-Access-Key":fir_r.text[1:-1]
            })
        cont_r=requests.get(
            f'https://damacacademy.damacgroup.com/modular/v1/api/module/{m_id}/content/file-path?activityIds={a_id}', 
            headers={
                "Learntron-Api-Token":sec_r.text[1:-1]
            })
        print(f"API Result : for m_id {m_id} and a_id {a_id} is ",cont_r.status_code)
        res_json = json.loads(cont_r.text)
        reslut_urls = [i['orginalFileUrl'] for i in res_json]
        return reslut_urls[0]
    except Exception as e:
        print(e)
        raise e
    


# Placeholder function. Replace with your actual function.
def get_video_details(collection, query):
    print("Getting video details ...")
    query = query
    resp = requests.get(os.getenv("MODEL_ENDPOINT"),json={'text': query})
    query_vector = json.loads(resp.json()['vector'])
    col = collection
    response = col.query.hybrid(
        query=query,
        vector=list(query_vector),
        limit=3,
        alpha=0.3,
        target_vector="chunks",  # Specify the target vector for named vector collections
        return_metadata=MetadataQuery(score=True, explain_score=True)
    )

    
    results = []
    for obj in response.objects:
        score = obj.metadata.score
        result_dict = obj.properties
        a_id = result_dict['act_id']
        chunk = result_dict['chunks']
        m_id =  result_dict['module_id']
        chunk_id = result_dict['chunk_id']
        start_time = float(result_dict['exacts_seconds'].split(",")[0][1:].strip())
        stop_time = float(result_dict['exacts_seconds'].split(",")[1][:-1].strip())

        results.append(
            {
                "activity_id":a_id,
                "module_id":m_id,
                "chunk_id":chunk_id,
                "chunk":chunk,
                "start_time":int(start_time),
                "stop_time":int(stop_time),
                "url":get_content_url(int(m_id), int(a_id)),
                "score":score
            }
        )
    print("Got video details ...")
    return results

def get_context(aid, chunk_id):    
    print("Getting context for Generation ...")
    response = ms_marco_minilm.query.fetch_objects(
        filters=(
            Filter.all_of([  # Combines the below with `&`
                Filter.by_property("act_id").equal(aid),
                Filter.by_property("chunk_id").greater_than(chunk_id-5),
                Filter.by_property("chunk_id").less_than(chunk_id+5)
            ])
        ),
        limit=5
    )
    context = []
    for o in response.objects:
        context.append(o.properties['chunks'])
    print("Got context for generation ...")
    return {
        "activity_id":aid,
        "context": " ".join(context)
    }


def generate(context, query):
    print("Generating the text output ...")
    messages = [
        {
            "role":"system", 
            "content": """
            Your job is to generate a response for a phrase or a question thats being searched in a search bar,
            I will provide you with the phrase or question asked in the search and also the content retrieved from a vector database
            the generated response should STRICTLY stick to the content provided and the generation should be a guiding / value adding information for the phrase or question
            going out of the content provided to generate is strictly prohibited
            """
        },
        {
            "role":"user" ,
            "content":f"""
            The query for which you have to generate a response for is ~~~{query}~~~, 
            The content from the Vector DB that the response must stictly adhere to is ~~~{context}~~~,
            Both the inputs are delimited by '~~~' triple tilde symbols
            Generete a response for the query 
            """
        }
    ]
    response = azure_client.chat.completions.create(model = "ChatGPT",messages = messages, temperature=0.5)
    print("Completed text output generation ...")
    return response.choices[0].message.content


def upload_to_bucket(blob_name, dataframe, bucket_name):

    print("Uploading the result into GCS bucket ...")

    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        'creds.json')

    #print(buckets = list(storage_client.list_buckets())

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(dataframe.to_csv(), 'text/csv')

    #returns a public url
    print("Upload completed ...")
    return blob.public_url

st.title("DAMAC - Deep Search")

query = st.text_input("Enter search query:")
search_button = st.button("Search")
feedback_dict = {
    "query":[],
    'text_chunk':[],
    "activity_id":[],
    "chunk_id":[],
    "response":[],
    "score":[]
}
if search_button and query:
    
    col = client.collections.get("Deep_search_Damac")
    video_details = get_video_details(col, query)
    
    context_results = []
    
    for idx, jsn in enumerate(video_details):

        activity_id = jsn['activity_id']
        chunk_id = jsn['chunk_id']
        text_chunk = jsn['chunk'] 
        start_time = jsn['start_time']
        end_time = jsn['stop_time']
        url = jsn['url']
        score = jsn['score']
        
        
        context_dict = get_context(activity_id, chunk_id)
        st.write(f"Result : {idx+1}")
        st.write(f"**Suggested clip:**")
        st.write(f"Playing the video from {start_time}s to {end_time}s:")
        
        st.video(url, start_time = start_time, end_time = end_time, autoplay = False, muted=True)
        st.write("Generated output : \n")
        generated_res = generate(context_dict['context'],query)
        st.write(generated_res)


        
        feedback_dict['text_chunk'].append(text_chunk)
        feedback_dict['query'].append(query)
        feedback_dict['response'].append(generated_res)
        feedback_dict['score'].append(score)
        feedback_dict["activity_id"].append(activity_id)
        feedback_dict["chunk_id"].append(chunk_id)
        


        st.write("_____________________________________________________________________________")

    
