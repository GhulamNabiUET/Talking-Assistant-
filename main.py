import streamlit as st
from audio_recorder_streamlit import audio_recorder
import whisper
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from gtts import gTTS

# Initialize models and data
@st.cache(allow_output_mutation=True)
def load_models():
    # Load Whisper model
    whisper_model = whisper.load_model("base")

    # Load and split documents
    pdf_path = "D:/Virtual Assistent/Runpod/07-Islamic-Adab-Good-Manners-or-Etiquette.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    # Create vector store
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma.from_documents(final_documents, hf_embeddings)

    # Set up RetrievalQA
    retriever = vectorstore.as_retriever()
    hf_pipeline = HuggingFacePipeline.from_model_id(
        model_id="gpt2",
        task="text-generation",
        pipeline_kwargs={"temperature": 0.1, "max_new_tokens": 200}
    )
    prompt_template = """
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context and in some descriptive way as well.

    {context}
    Question: {question}

    Helpful Answers:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return whisper_model, retrievalQA

# Load models and pipeline
whisper_model, retrievalQA = load_models()

# Function to transcribe audio using Whisper model
def transcribe_audio(audio_bytes):
    audio_path = "audio_file1.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_path = "response.mp3"
    tts.save(audio_path)
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    return audio_bytes

# Streamlit App Title and Description
st.title("🧑‍💻 Ashal 🌟✨🌍 Assistant 🤖")

"""
Hi🤖 just click on the voice recorder and let me know how I can help you today?
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["initialized"] = False  # Add an initialized flag

# Force a rerun if not initialized
if not st.session_state["initialized"]:
    st.session_state["initialized"] = True
    st.experimental_rerun()

# Capture user input through audio recorder
audio_bytes = audio_recorder(key="recorder")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.audio(message["audio"], format="audio/wav")
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.audio(message["audio"], format="audio/mp3")

if audio_bytes:
    with st.spinner('Processing...'):
        # Transcribe the audio file to text using Whisper model
        transcribed_text = transcribe_audio(audio_bytes)
        st.success("Audio file successfully transcribed!")

        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": transcribed_text, "audio": audio_bytes})

        # Use the transcribed text as the query for the RetrievalQA pipeline
        result = retrievalQA.invoke({"query": transcribed_text})
        answer = result['result']

        # Convert the assistant's response to speech
        response_audio_bytes = text_to_speech(answer)

        # Append assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer, "audio": response_audio_bytes})

        # Display the transcribed text and the assistant's response
        with st.chat_message("user"):
            st.markdown(transcribed_text)
            st.audio(audio_bytes, format="audio/wav")

        with st.chat_message("assistant"):
            # st.markdown(answer)
            st.audio(response_audio_bytes, format="audio/mp3")

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
