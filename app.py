from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import  FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import streamlit as st 
def get_video_id(url):
    paser_url=urlparse(url)
    if paser_url.hostname=="www.youtube.com":
        return parse_qs(paser_url.query).get("v",[None])[0]
    elif paser_url.hostname == "youtu.be":
        return paser_url.path[1:]
    return None
# # ----streamlit--------
st.set_page_config(page_title="youtube chat box")
st.title("write any query about any  youtube video ")
load_dotenv()
 #----------model---------
llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task='text_generation'
)
model=ChatHuggingFace(llm=llm)
#-----------promts-----------
promt=PromptTemplate(
    template="""your the very hepful assitance 
    answer the question from only provide transcripted text 
    if you dont know the answer just say i dont know 
    text={text}
    questioned={question}""",
    input_variables=['text','question']
)
youtube_url = st.text_input("Enter YouTube URL")
video_id = None
if youtube_url:
    video_id = get_video_id(youtube_url)
# -----------text_splitter--------------
@st.cache_resource
def create_vectore_store(video_id):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
# ---------loading_youtube_transcript-----------
    ytt=YouTubeTranscriptApi()
    try:
        transcript_text=ytt.fetch(video_id=video_id,languages=['en']) 
        transcript_text=" ".join([item.text for item in  transcript_text])
        chunks=splitter.split_text(transcript_text)
    except Exception as e:
        st.error(f"could not find the transcript {e}")
        st.stop()
        
#----------crating_embedding----------
    Embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
#---------creating_vector_database-------------
    vector_store=FAISS.from_texts(
        texts=chunks,
        embedding=Embedding)
    return vector_store
if video_id:
    vector_store=create_vectore_store(video_id)
    retriver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)
# -----------------input_loop_--------------
    question=st.text_input("ask a questione ")
    if st.button("ask"):
        retrive_text=retriver.invoke(question)
        context_text="\n\n".join(doc.page_content for doc in retrive_text)
        final_promt=promt.invoke({'text':context_text,'question':question})
        with st.spinner("searching........."):
            final_llm=model.invoke(final_promt)
        st.subheader("answer")
        st.write(final_llm.content)
        