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

# 🎨 Modern Mobile-Inspired CSS
st.markdown("""
<style>
    /* Reset and base styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding: 0 !important;
        max-width: 480px !important;
        margin: 0 auto;
    }
    
    /* Header */
    .chat-header {
        background: white;
        padding: 20px 20px 15px 20px;
        border-radius: 0 0 25px 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    
    .header-title {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 22px;
        font-weight: 600;
        color: #2d3748;
    }
    
    .header-icon {
        font-size: 28px;
    }
    
    .header-buttons {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    
    .lang-btn {
        background: #4299e1;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        border: none;
        cursor: pointer;
    }
    
    .delete-btn {
        background: #f7fafc;
        color: #718096;
        padding: 8px 12px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        font-size: 18px;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 25px;
        padding: 20px;
        margin: 0 15px;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    /* Message bubbles */
    .message-wrapper {
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
    }
    
    .user-message-wrapper {
        align-items: flex-end;
    }
    
    .bot-message-wrapper {
        align-items: flex-start;
    }
    
    .message-label {
        font-size: 12px;
        color: #718096;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 6px;
        font-weight: 500;
    }
    
    .message-bubble {
        padding: 15px 18px;
        border-radius: 20px;
        max-width: 80%;
        word-wrap: break-word;
        line-height: 1.5;
        font-size: 15px;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 5px;
    }
    
    .bot-bubble {
        background: #f7fafc;
        color: #2d3748;
        border-bottom-left-radius: 5px;
        border: 1px solid #e2e8f0;
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 480px;
        background: white;
        padding: 15px 20px 25px 20px;
        border-radius: 25px 25px 0 0;
        box-shadow: 0 -4px 6px rgba(0,0,0,0.1);
    }
    
    .stChatInput {
        border-radius: 25px !important;
    }
    
    /* Streamlit chat message overrides */
    .stChatMessage {
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    
    /* Loading animation */
    .thinking-dots {
        display: inline-flex;
        gap: 4px;
    }
    
    .thinking-dots span {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .thinking-dots span:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .thinking-dots span:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
        }
        40% {
            transform: scale(1);
        }
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: white;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px;
        font-weight: 600;
        font-size: 16px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6b4193 100%);
    }
</style>
""", unsafe_allow_html=True)

# 📱 Header
st.markdown("""
<div class="chat-header">
    <div class="header-title">
        <span class="header-icon">🤖</span>
        <span>Buddy AI</span>
    </div>
    <div class="header-buttons">
        <button class="lang-btn">हिंदी</button>
    </div>
</div>
""", unsafe_allow_html=True)

# 📂 LOAD PDFs SAFELY
DATA_PATH = os.path.join(os.getcwd(), "data")
all_docs = []

if not os.path.exists(DATA_PATH):
    st.error("❌ 'data' folder not found. Upload PDFs to the data directory.")
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

Use the conversation history and provided context to answer naturally and professionally.
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

# 💬 Chat display container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display previous chat with custom styling
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
        <div class="message-wrapper user-message-wrapper">
            <div class="message-label">
                <span>You</span>
            </div>
            <div class="message-bubble user-bubble">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-wrapper bot-message-wrapper">
            <div class="message-label">
                🤖 <span>Buddy</span>
            </div>
            <div class="message-bubble bot-bubble">
                {message["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ⌨️ User input
if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show thinking animation
    with st.spinner(""):
        history_text = format_history(st.session_state.messages[-6:])
        answer = rag_chain.invoke({
            "question": user_input,
            "history": history_text
        })
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ⚙️ Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.info("💡 **Buddy AI**\n\nAsk me anything about:\n- Your documents\n- Company information\n- Policies & procedures\n- Any questions you have!")
    
    st.divider()
    
    st.caption("☁️ Powered by Pinecone Vector Database")
    st.caption("🤖 Using OpenAI GPT-4o-mini")
