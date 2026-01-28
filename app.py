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

# 💬 Initialize chat memory EARLY
if "messages" not in st.session_state:
    st.session_state.messages = []

# 🌙 Page config
st.set_page_config(page_title="Buddy AI", page_icon="🤖", layout="wide")

st.markdown("""<style>/* YOUR CSS */</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1 class="logo">🤖 Buddy AI</h1>
</div>
""", unsafe_allow_html=True)

# 📂 LOAD PDFs SAFELY
DATA_PATH = os.path.join(os.getcwd(), "data")
all_docs = []

if not os.path.exists(DATA_PATH):
    st.error("❌ 'data' folder not found. Upload PDFs to GitHub.")
    st.stop()

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        all_docs.extend(loader.load())

if not all_docs:
    st.error("❌ No PDFs found in `data` folder.")
    st.stop()

# ✂️ Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
docs = text_splitter.split_documents(all_docs)

# 🧠 Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 🌲 Pinecone Setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("buddy-ai-index")

vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 🚀 Upload to Pinecone only once
stats = index.describe_index_stats()
if stats["total_vector_count"] == 0:
    with st.spinner("🔄 First-time setup: Uploading documents to Pinecone..."):
        vectorstore.add_documents(docs)
    st.success("✅ Documents uploaded!")
else:
    st.info("📦 Using existing Pinecone index")

# 🤖 LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 🧠 History formatter
def format_history(messages):
    return "\n".join(
        f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
        for m in messages
    )

# 📄 Document formatter
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 🧾 Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are Buddy AI, a helpful assistant answering questions from company documents.

Use the conversation history and provided context to answer naturally.
If the answer is not found, give the closest helpful answer.
Only say "I don't know" if completely unrelated.

Chat History:
{history}

Context:
{context}

User Question:
{question}
""")

# 🔗 RAG Chain
rag_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "history": lambda x: x["history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 💬 Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# ⌨️ User input
if user_input := st.chat_input("Ask about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            history_text = format_history(st.session_state.messages[-6:])
            answer = rag_chain.invoke({
                "question": user_input,
                "history": history_text
            })

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ⚙️ Sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.info("☁️ Using Pinecone Vector Database")
