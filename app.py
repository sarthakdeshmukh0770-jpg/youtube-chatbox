from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import  FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import streamlit as st 
from yt_dlp import YoutubeDL
def get_youtube_info(url):
    ydl_opts={
        'quiet':True,
        'skip_download':True
    }
    with YoutubeDL(ydl_opts) as yd:
        info=yd.extract_info(url,download=False)
    return {
        "title": info.get("title"),
        "channel": info.get("channel"),
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration")
    }

def get_video_id(url):
    paser_url=urlparse(url)
    if paser_url.hostname=="www.youtube.com":
        return parse_qs(paser_url.query).get("v",[None])[0]
    elif paser_url.hostname == "youtu.be":
        return paser_url.path[1:]
    return None
if "messages" not in st.session_state:
    st.session_state.messages=[]
st.title("🎥 YouTube Video ChatBot")
st.caption("Ask anything about a YouTube video's transcript using AI.")
st.caption(" paste your youtube url in the sidebar press the left-top most >> button ")
# ---------------- Sidebar ----------------

with st.sidebar:
    st.set_page_config(
    page_title="YouTube ChatBot",
    page_icon="🎥",
    layout="wide"
)
    st.title("🎥 YouTube ChatBot")
    st.markdown("---")
    youtube_url = st.text_input(
        "📺 YouTube URL",
        placeholder="Paste YouTube URL..."
    )
    video_info = None
    if youtube_url:
        try:
            video_info = get_youtube_info(youtube_url)
            st.image(
                video_info["thumbnail"],
                use_container_width=True
            )
            st.markdown(f"### {video_info['title']}")
            st.write(f"**👤 Channel:** {video_info['channel']}")
            minutes = video_info["duration"] // 60
            seconds = video_info["duration"] % 60
            st.write(f"**⏱ Duration:** {minutes}:{seconds:02d}")
            st.success("✅ Transcript Ready")
        except Exception:
            st.error("Unable to load video information.")
    st.markdown("---")
    st.subheader("📊 Project Information")
    st.metric(
        "💬 Messages",
        len(st.session_state.messages)
    )
    st.write("**🧠 Embedding Model**")
    st.caption("sentence-transformers/all-MiniLM-L6-v2")
    st.write("**🤖 LLM**")
    st.caption("Qwen/Qwen2.5-7B-Instruct")
    st.write("**🗄 Vector Store**")
    st.caption("FAISS")
    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; font-size:14px; color:gray;">
             <b>🎥 YouTube RAG ChatBot</b><br><br>

            👨‍💻 Developed by
            Sarthak Deshmukh""",
        unsafe_allow_html=True,
    )
# # ----streamlit--------
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
    conversion_history={history}
    text={text}
    questioned={question}""",
    input_variables=['text','question','history']
)

# -----------text_splitter--------------
@st.cache_resource
def create_vectore_store(video_id):
    progress = st.progress(0)
    status = st.empty()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
# ---------loading_youtube_transcript-----------
    status.text("📄 Fetching YouTube transcript...")
    progress.progress(20)
    ytt=YouTubeTranscriptApi()
    try:
        transcript_text=ytt.fetch(video_id=video_id,languages=['en']) 
        status.text("✂️ Splitting transcript into chunks...")
        progress.progress(45)
        transcript_text=" ".join([item.text for item in  transcript_text])
        chunks=splitter.split_text(transcript_text)
    except Exception as e:
        st.error(f"could not find the transcript {e}")
        st.stop()
        
#----------crating_embedding----------
    status.text("🧠 Creating embeddings...")
    progress.progress(70)
    Embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
#---------creating_vector_database-------------
    status.text("💾 Building FAISS vector database...")
    progress.progress(90)
    vector_store=FAISS.from_texts(
        texts=chunks,
        embedding=Embedding)
    progress.progress(100)
    status.success("✅ Video processed successfully!")
    import time
    time.sleep(1)
    progress.empty()
    status.empty()
    return vector_store
video_id = None

if youtube_url:
    video_id = get_video_id(youtube_url)
if video_id:
    vector_store=create_vectore_store(video_id)
    retriver=vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":5}
)
# -----------------input_loop_--------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question=st.chat_input("ask a questione ")
history=" "
if question and video_id:
        for msg in st.session_state.messages:
            history+=f"{msg['role']}:{msg['content']}\n"
        st.chat_message('user').markdown(question)
        st.session_state.messages.append(
            {
                'role':"user",
                'content':question
            }
        )
        retrive_text=retriver.invoke(question)
        context_text="\n\n".join(doc.page_content for doc in retrive_text)
        final_promt=promt.invoke({'text':context_text,'question':question,'history':history})
        with st.spinner("searching........."):
            final_llm=model.invoke(final_promt)
        st.chat_message("assistant").markdown(final_llm.content)
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":final_llm.content
            }
        )

        
        