"""
Application Transition Assistant - Version Cloud
Adaptée pour fonctionner sur Streamlit Cloud avec Hugging Face (gratuit)
"""

import streamlit as st
import os
import json
from pathlib import Path
# Essayer d'importer GoogleDriveLoader
try:
    from langchain_google_community import GoogleDriveLoader
except ImportError:
    st.error("❌ Le module 'langchain-google-community' n'est pas installé. "
             "Ajoutez-le dans requirements.txt :\n\nlangchain-google-community")
    GoogleDriveLoader = None  # fallback pour éviter le crash

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time
from datetime import datetime

# --- CONFIGURATION CLOUD ---

# Détection de l'environnement
IS_CLOUD = 'STREAMLIT_CLOUD' in os.environ or 'google_credentials' in st.secrets

if IS_CLOUD:
    # Configuration depuis Streamlit Secrets
    if 'google_credentials' in st.secrets:
        # Créer le fichier credentials.json depuis les secrets
        creds_dict = dict(st.secrets['google_credentials'])
        with open('credentials.json', 'w') as f:
            json.dump(creds_dict, f)
    
    # Récupérer les configurations
    FOLDER_ID = st.secrets.get('app_config', {}).get('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = st.secrets.get('app_config', {}).get('HUGGINGFACE_TOKEN', None)
else:
    # Configuration locale
    FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', None)

SERVICE_ACCOUNT_FILE = "credentials.json"

# --- CUSTOM CSS (identique) ---
def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
        
        :root {
            --gold: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
            --silver: linear-gradient(135deg, #C0C0C0 0%, #E5E5E5 50%, #C0C0C0 100%);
            --bronze: linear-gradient(135deg, #CD7F32 0%, #B87333 50%, #CD7F32 100%);
            --glass: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --shadow-3d: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 50%, #16213e 100%);
            position: relative;
            overflow: hidden;
        }
        
        .stApp::before {
            content: "";
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 20% 80%, rgba(255, 215, 0, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(192, 192, 192, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(205, 127, 50, 0.1) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
            pointer-events: none;
        }
        
        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(-20px, -20px) rotate(120deg); }
            66% { transform: translate(20px, -10px) rotate(240deg); }
        }
        
        .main-header {
            background: var(--glass);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-3d);
            position: relative;
            transform-style: preserve-3d;
            animation: float-3d 6s ease-in-out infinite;
        }
        
        @keyframes float-3d {
            0%, 100% { transform: translateY(0px) rotateX(0deg); }
            50% { transform: translateY(-10px) rotateX(2deg); }
        }
        
        .main-title {
            font-family: 'Orbitron', sans-serif;
            font-weight: 900;
            font-size: 3rem;
            background: var(--gold);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 0 4px 8px rgba(255, 215, 0, 0.3);
            margin: 0;
        }
        
        .subtitle {
            font-family: 'Exo 2', sans-serif;
            font-weight: 300;
            color: #C0C0C0;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        
        .medals-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .medal {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            animation: rotate-medal 8s linear infinite;
            transform-style: preserve-3d;
        }
        
        .medal-gold { background: var(--gold); }
        .medal-silver { background: var(--silver); }
        .medal-bronze { background: var(--bronze); }
        
        @keyframes rotate-medal {
            0% { transform: rotateY(0deg); }
            100% { transform: rotateY(360deg); }
        }
        
        .stChatMessage {
            background: var(--glass) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 16px !important;
            margin: 0.5rem 0 !important;
            padding: 1rem !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stChatInput > div > div > input {
            background: var(--glass) !important;
            backdrop-filter: blur(10px) !important;
            border: 2px solid transparent !important;
            background-image: linear-gradient(var(--glass), var(--glass)), 
                              var(--gold) !important;
            background-origin: border-box !important;
            background-clip: padding-box, border-box !important;
            border-radius: 12px !important;
            color: white !important;
            font-family: 'Exo 2', sans-serif !important;
        }
        
        @media (max-width: 768px) {
            .main-title { font-size: 2rem; }
            .subtitle { font-size: 1rem; }
            .medal { width: 40px; height: 40px; font-size: 1.5rem; }
        }
        
        @media (max-width: 480px) {
            .main-title { font-size: 1.5rem; }
            .subtitle { font-size: 0.9rem; }
        }
    </style>
    """

# --- INTERFACE STREAMLIT ---
st.set_page_config(
    page_title="Transition Assistant | Elite Athletes",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
st.markdown(load_css(), unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">TRANSITION ASSISTANT</h1>
        <p class="subtitle">Elite Athletes Career Evolution Platform</p>
        <div class="medals-container">
            <div class="medal medal-gold">🥇</div>
            <div class="medal medal-silver">🥈</div>
            <div class="medal medal-bronze">🥉</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- CONFIGURATION DU MODÈLE ---

@st.cache_resource
def get_llm():
    """
    Configure le modèle LLM selon l'environnement (cloud ou local)
    """
    if HUGGINGFACE_TOKEN:
        # Version Cloud avec Hugging Face (GRATUIT)
        try:
            llm = HuggingFaceHub(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                model_kwargs={
                    "temperature": 0.1,
                    "max_new_tokens": 512,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1
                }
            )
            st.success("✅ Modèle cloud Hugging Face connecté")
            return llm
        except Exception as e:
            st.error(f"❌ Erreur Hugging Face: {str(e)}")
            st.info("💡 Créez un token gratuit sur https://huggingface.co/settings/tokens")
            return None
    else:
        # Essayer Ollama en local
        try:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="mistral", temperature=0.1)
            st.success("✅ Modèle local Ollama connecté")
            return llm
        except:
            st.warning("⚠️ Ni Hugging Face ni Ollama configurés")
            
            # Instructions pour configurer
            with st.expander("📋 Comment configurer un modèle IA gratuit?"):
                st.markdown("""
                ### Option 1: Hugging Face (Recommandé pour le cloud)
                
                1. Créez un compte sur [huggingface.co](https://huggingface.co)
                2. Allez dans Settings > Access Tokens
                3. Créez un token "Read"
                4. Ajoutez-le dans Streamlit Secrets:
                ```toml
                [app_config]
                HUGGINGFACE_TOKEN = "hf_votre_token_ici"
                ```
                
                ### Option 2: Ollama (Pour usage local)
                
                1. Installez [Ollama](https://ollama.ai)
                2. Lancez: `ollama run mistral`
                """)
            return None

# --- FONCTIONS PRINCIPALES ---

@st.cache_resource
def initialize_knowledge_base():
    """
    Charge les documents du Google Drive et crée l'index de recherche
    """
    if GoogleDriveLoader is None:
        st.warning("⚠️ GoogleDriveLoader indisponible. Vérifiez votre requirements.txt.")
    return None
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"⚠️ Fichier '{SERVICE_ACCOUNT_FILE}' introuvable")
        
        with st.expander("📋 Comment configurer Google Drive?"):
            st.markdown("""
            1. **Google Cloud Console**: Créez un projet
            2. **Activez l'API Drive**
            3. **Créez un compte de service**
            4. **Téléchargez le JSON** → `credentials.json`
            5. **Partagez votre dossier** avec l'email du compte
            6. **Copiez l'ID du dossier** dans les secrets
            """)
        return None

    progress_placeholder = st.empty()
    
    with progress_placeholder.container():
        with st.spinner("🔄 Chargement de la base de connaissances..."):
            try:
                # Chargement depuis Google Drive
                loader = GoogleDriveLoader(
                    folder_id=FOLDER_ID,
                    file_types=["docx", "doc"],
                    service_account_key=SERVICE_ACCOUNT_FILE,
                    recursive=True
                )
                docs = loader.load()
                
                if not docs:
                    st.warning("📂 Aucun document trouvé")
                    return None

                # Découpage
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Embeddings gratuits
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'}
                )
                
                # Vectorstore
                vectorstore = FAISS.from_documents(splits, embeddings)
                
                progress_placeholder.empty()
                st.success(f"✅ {len(docs)} documents chargés!")
                
                return vectorstore

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Erreur: {str(e)}")
                return None

# --- INITIALISATION ---

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Obtenir le modèle LLM
    llm = get_llm()
    
    if llm:
        # Initialiser la base de connaissances
        vectorstore = initialize_knowledge_base()
        
        if vectorstore:
            # Configuration RAG
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            response = qa_chain.run(user_input)
            answer = qa_chain.run(user_input)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            # Prompt système
            system_prompt = """
            You are an expert assistant for elite athlete career transition.
            
            INSTRUCTIONS:
            1. Answer ONLY based on the provided context
            2. Detect the language (French/English) and respond in the SAME language
            3. Be professional and empathetic
            
            Context: {context}
            Question: {input}
            """
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )

            
            # --- CHAT INTERFACE ---
            
            if "messages" not in st.session_state:
                st.session_state.messages = []
                welcome = """
                🌟 **Bienvenue / Welcome!**
                
                Je suis votre assistant spécialisé dans la transition de carrière.
                I am your specialized career transition assistant.
                
                💬 Posez vos questions / Ask your questions!
                """
                st.session_state.messages.append({"role": "assistant", "content": welcome})
            
            # Afficher l'historique
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Zone de saisie
            if user_input := st.chat_input("💬 Votre question / Your question..."):
                # Afficher la question
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Générer la réponse
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Réflexion..."):
                        try:
                            response = rag_chain.invoke({"input": user_input})
                            answer = response["answer"]
                        except Exception as e:
                            answer = f"❌ Erreur: {str(e)}"
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("📂 Configurez Google Drive pour commencer")
    else:
        st.warning("🤖 Configurez un modèle IA pour commencer")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #C0C0C0;">
        <p style="font-family: 'Exo 2', sans-serif; font-size: 0.9rem;">
            🏅 Transition Assistant © 2024 | Cloud Edition
        </p>
    </div>
""", unsafe_allow_html=True)
