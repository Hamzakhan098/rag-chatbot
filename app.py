from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# 🌙 ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Buddy AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 ---------- UI CSS (kept same) ----------
st.markdown("""<style>
/* YOUR EXISTING CSS */
</style>""", unsafe_allow_html=True)

# 🖤 ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1 class="logo">🤖 Buddy AI</h1>
</div>
""", unsafe_allow_html=True)

# 💬 ---------- CHAT MEMORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 📂 ---------- LOAD PDFs ----------
DATA_PATH = os.path.join(os.getcwd(), "data")
all_docs = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        all_docs.extend(loader.load())

if not all_docs:
    st.error("❌ No PDFs found in `data` folder. Add PDFs and rerun.")
    st.stop()

# ✂️ ---------- IMPROVED CHUNKING ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
docs = text_splitter.split_documents(all_docs)

# 🧠 ---------- EMBEDDINGS ----------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 🌲 ---------- PINECONE CONNECTION ----------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("buddy-ai-index")

vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 🚀 ---------- UPLOAD TO PINECONE IF EMPTY ----------
if index.describe_index_stats()["total_vector_count"] == 0:
    with st.spinner("Uploading documents to Pinecone (first time setup)..."):
        vectorstore.add_documents(docs)
    st.success("Documents uploaded to Pinecone!")

# 🤖 ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🧠 ---------- MEMORY FORMATTER ----------
def format_history(messages):
    history_text = ""
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{role}: {m['content']}\n"
    return history_text

# 📄 ---------- DOC FORMATTER ----------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🧾 ---------- SMARTER PROMPT ----------
prompt = ChatPromptTemplate.from_template("""
You are Buddy AI, a helpful assistant answering questions from company documents.

Use the conversation history and provided context to answer naturally.
If the exact answer is not found, give the closest helpful answer based on context.
Only say "I don't know" if the context is completely unrelated.

Chat History:
{history}

Context:
{context}

User Question:
{question}
""")

# 🔗 ---------- RAG PIPELINE WITH MEMORY ----------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x,
        "history": lambda x: format_history(st.session_state.messages[-6:])
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 💬 ---------- DISPLAY CHAT ----------
for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="👤" if message["role"] == "user" else "🤖"
    ):
        st.markdown(message["content"])

# ⌨️ ---------- INPUT ----------
if user_input := st.chat_input("Ask about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            docs = retriever.invoke(user_input)
            answer = rag_chain.invoke(user_input)

        st.markdown(answer)

        with st.expander("📄 Sources"):
            for i, doc in enumerate(docs):
                st.write(f"Source {i+1} — Page {doc.metadata.get('page', 'N/A')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ⚙️ ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.info("☁️ Using Pinecone Vector Database")
