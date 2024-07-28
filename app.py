import streamlit as st
from audio_recorder_streamlit import audio_recorder
import whisper
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from gtts import gTTS
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

@st.cache(allow_output_mutation=True)
def load_models():
    # Load Whisper model
    whisper_model = whisper.load_model("base")

    # Load and split documents
    loader = PyPDFDirectoryLoader('C:/Users/USER/Virtual Assistant/GROQ/knowledge_base')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)

    # Create vector store
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma.from_documents(final_documents, hf_embeddings)

    # Set up retriever
    retriever = vectorstore.as_retriever()

    # Initialize the ChatGroq LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Set up retrieval chain with history awareness
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return whisper_model, conversational_rag_chain

# Load models and pipeline
whisper_model, conversational_rag_chain = load_models()

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

        # Use the transcribed text as the query for the RAG pipeline
        result = conversational_rag_chain.invoke(
            {"input": transcribed_text, "chat_history": st.session_state.messages},
            config={"configurable": {"session_id": "abc123"}}
        )
        extracted_answer = result["answer"]

        # Convert the assistant's response to speech
        response_audio_bytes = text_to_speech(extracted_answer)

        # Append assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": extracted_answer, "audio": response_audio_bytes})

        # Display the transcribed text and the assistant's response
        with st.chat_message("user"):
            st.markdown(transcribed_text)
            st.audio(audio_bytes, format="audio/wav")

        with st.chat_message("assistant"):
            st.markdown(extracted_answer)
            st.audio(response_audio_bytes, format="audio/mp3")

# Option to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
