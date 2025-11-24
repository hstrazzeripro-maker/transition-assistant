import streamlit as st
import os
from pathlib import Path
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import time
from datetime import datetime

# --- CONFIGURATION ---
FOLDER_ID = "VOTRE_FOLDER_ID_ICI"  # Remplacez par votre ID de dossier Google Drive
SERVICE_ACCOUNT_FILE = "credentials.json"

# --- CUSTOM CSS DESIGN ---
def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
        
        /* Variables CSS pour les couleurs métalliques */
        :root {
            --gold: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
            --silver: linear-gradient(135deg, #C0C0C0 0%, #E5E5E5 50%, #C0C0C0 100%);
            --bronze: linear-gradient(135deg, #CD7F32 0%, #B87333 50%, #CD7F32 100%);
            --glass: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --shadow-3d: 0 20px 40px rgba(0, 0, 0, 0.3);
        }
        
        /* Background moderne avec effet parallaxe */
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
        
        /* Header avec effet glass morphism 3D */
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
            position: relative;
        }
        
        .subtitle {
            font-family: 'Exo 2', sans-serif;
            font-weight: 300;
            color: #C0C0C0;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        
        /* Médailles flottantes */
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
        
        /* Chat container avec effet glass */
        .stChatMessage {
            background: var(--glass) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 16px !important;
            margin: 0.5rem 0 !important;
            padding: 1rem !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatMessage:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Input field avec style moderne */
        .stChatInput > div > div > input {
            background: var(--glass) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border: 2px solid transparent !important;
            background-image: linear-gradient(var(--glass), var(--glass)), 
                              var(--gold) !important;
            background-origin: border-box !important;
            background-clip: padding-box, border-box !important;
            border-radius: 12px !important;
            color: white !important;
            font-family: 'Exo 2', sans-serif !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInput > div > div > input:focus {
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3) !important;
            transform: translateY(-2px);
        }
        
        .stChatInput > div > div > input::placeholder {
            color: rgba(192, 192, 192, 0.6) !important;
        }
        
        /* Buttons avec effet métallique */
        .stButton > button {
            background: var(--gold) !important;
            color: #1a1a2e !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-family: 'Exo 2', sans-serif !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 16px rgba(255, 215, 0, 0.3) !important;
            transition: all 0.3s ease !important;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: "";
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #FFD700, #FFA500, #FFD700);
            border-radius: 12px;
            opacity: 0;
            z-index: -1;
            transition: opacity 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 8px 24px rgba(255, 215, 0, 0.4) !important;
        }
        
        .stButton > button:hover::before {
            opacity: 1;
        }
        
        /* Spinner personnalisé */
        .stSpinner > div {
            border-color: #FFD700 !important;
        }
        
        /* Toast notifications */
        .stToast {
            background: var(--glass) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid var(--glass-border) !important;
            border-radius: 12px !important;
            color: white !important;
            font-family: 'Exo 2', sans-serif !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
            
            .subtitle {
                font-size: 1rem;
            }
            
            .medal {
                width: 40px;
                height: 40px;
                font-size: 1.5rem;
            }
            
            .medals-container {
                gap: 1rem;
            }
            
            .main-header {
                padding: 1.5rem;
                border-radius: 16px;
            }
        }
        
        @media (max-width: 480px) {
            .main-title {
                font-size: 1.5rem;
            }
            
            .subtitle {
                font-size: 0.9rem;
            }
            
            .stChatMessage {
                padding: 0.75rem !important;
                border-radius: 12px !important;
            }
        }
        
        /* Loading animation */
        .loading-dots {
            display: inline-block;
            position: relative;
            width: 80px;
            height: 20px;
        }
        
        .loading-dots div {
            position: absolute;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: var(--gold);
            animation: loading-dots 1.4s infinite ease-in-out both;
        }
        
        .loading-dots div:nth-child(1) {
            left: 8px;
            animation-delay: -0.32s;
        }
        
        .loading-dots div:nth-child(2) {
            left: 32px;
            animation-delay: -0.16s;
        }
        
        .loading-dots div:nth-child(3) {
            left: 56px;
            animation-delay: 0;
        }
        
        @keyframes loading-dots {
            0%, 80%, 100% {
                transform: scale(0.8);
                background: var(--bronze);
            }
            40% {
                transform: scale(1.2);
                background: var(--gold);
            }
        }
        
        /* Effet de particules flottantes */
        .particle {
            position: fixed;
            pointer-events: none;
            opacity: 0.3;
            animation: particle-float 15s infinite;
        }
        
        @keyframes particle-float {
            0% {
                transform: translateY(100vh) translateX(0) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 0.3;
            }
            90% {
                opacity: 0.3;
            }
            100% {
                transform: translateY(-10vh) translateX(100px) rotate(720deg);
                opacity: 0;
            }
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

# Injection du CSS personnalisé
st.markdown(load_css(), unsafe_allow_html=True)

# Header avec design glass morphism
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

# Particules flottantes pour effet 3D
for i in range(5):
    st.markdown(f"""
        <div class="particle" style="
            left: {i * 20}%;
            animation-delay: {i * 3}s;
            font-size: {20 + i * 5}px;
        ">✨</div>
    """, unsafe_allow_html=True)

# --- FONCTIONS PRINCIPALES ---

@st.cache_resource
def initialize_knowledge_base():
    """
    Charge les documents du Google Drive et crée l'index de recherche local.
    """
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"⚠️ Fichier '{SERVICE_ACCOUNT_FILE}' introuvable. Veuillez configurer votre compte de service Google.")
        st.info("💡 Créez un compte de service dans Google Cloud Console et téléchargez le fichier JSON.")
        return None

    progress_container = st.empty()
    
    with progress_container.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; padding: 2rem;">
                    <div class="loading-dots">
                        <div></div>
                        <div></div>
                        <div></div>
                    </div>
                    <p style="color: #C0C0C0; margin-top: 1rem; font-family: 'Exo 2', sans-serif;">
                        Initialisation de la base de connaissances...
                    </p>
                </div>
            """, unsafe_allow_html=True)

    try:
        # 1. Chargement depuis Google Drive
        loader = GoogleDriveLoader(
            folder_id=FOLDER_ID,
            file_types=["docx", "doc"],
            service_account_key=SERVICE_ACCOUNT_FILE,
            recursive=True
        )
        docs = loader.load()
        
        if not docs:
            progress_container.empty()
            st.warning("📂 Aucun document trouvé dans le dossier 'Transition'.")
            return None

        # 2. Découpage des documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        # 3. Embeddings gratuits avec HuggingFace
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 4. Création du vectorstore FAISS
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        progress_container.empty()
        
        # Animation de succès
        success_placeholder = st.empty()
        with success_placeholder.container():
            st.success(f"✅ Base de connaissances initialisée avec {len(docs)} documents!")
            time.sleep(2)
        success_placeholder.empty()
        
        return vectorstore

    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Erreur lors de l'initialisation : {str(e)}")
        st.info("💡 Vérifiez votre configuration Google Drive et Ollama.")
        return None

# --- DÉTECTION DE LANGUE ---
def detect_language(text):
    """
    Détecte si le texte est en français ou en anglais.
    """
    french_words = ['le', 'la', 'les', 'un', 'une', 'de', 'du', 'et', 'est', 'que', 'qui', 'dans', 'pour', 'sur', 'avec']
    english_words = ['the', 'a', 'an', 'is', 'are', 'in', 'on', 'at', 'for', 'with', 'and', 'or', 'but', 'to', 'of']
    
    text_lower = text.lower()
    french_count = sum(1 for word in french_words if word in text_lower.split())
    english_count = sum(1 for word in english_words if word in text_lower.split())
    
    return "fr" if french_count > english_count else "en"

# --- INITIALISATION ---

# Colonnes pour la mise en page
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Vérification Ollama
    with st.expander("⚙️ Configuration système", expanded=False):
        st.markdown("""
        ### Prérequis :
        1. **Ollama** : Téléchargez sur [ollama.ai](https://ollama.ai)
        2. **Modèle Mistral** : Exécutez `ollama run mistral` dans votre terminal
        3. **Google Drive** : Configurez votre compte de service et l'ID du dossier
        4. **Python packages** : `pip install -r requirements.txt`
        """)
        
        # Test de connexion Ollama
        try:
            test_llm = ChatOllama(model="mistral", temperature=0)
            st.success("✅ Ollama connecté avec succès")
        except:
            st.error("❌ Ollama non détecté. Lancez `ollama run mistral` dans votre terminal.")

    # Initialisation de la base de connaissances
    vectorstore = initialize_knowledge_base()

    if vectorstore:
        # Configuration du RAG Chain
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Modèle Ollama local (gratuit)
        llm = ChatOllama(
            model="mistral",
            temperature=0,
            num_predict=2048
        )

        # Prompt système bilingue
        system_prompt = """
        You are an expert assistant specialized in elite athlete career transition. 
        You have access to a comprehensive knowledge base about athlete career evolution.
        
        STRICT INSTRUCTIONS:
        1. Answer ONLY based on the provided context. If information is not in the context, clearly state it.
        2. DETECT the language of the question (French or English) and RESPOND in the SAME language.
        3. Be professional, empathetic, and provide actionable insights.
        4. Structure your responses clearly with relevant examples when available.
        
        Context from the knowledge base:
        {context}
        
        Human Question: {input}
        
        Your Response (in the same language as the question):
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # --- INTERFACE CHAT ---
        
        # Initialisation de l'historique
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Message de bienvenue bilingue
            welcome_message = """
            🌟 **Bienvenue / Welcome!**
            
            Je suis votre assistant spécialisé dans la transition de carrière des athlètes de haut niveau.
            I am your specialized assistant for elite athlete career transition.
            
            💬 Posez vos questions en français ou en anglais / Ask your questions in French or English.
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})

        # Container pour les messages avec scroll
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Zone de saisie
        user_input = st.chat_input("💬 Posez votre question / Ask your question...")

        if user_input:
            # Ajout du message utilisateur
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)

            # Génération de la réponse
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                
                # Animation de réflexion
                with thinking_placeholder.container():
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col2:
                        st.markdown("""
                            <div style="text-align: center;">
                                <div class="loading-dots">
                                    <div></div>
                                    <div></div>
                                    <div></div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Traitement de la requête
                try:
                    response = rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    
                    # Ajout de métadonnées si disponibles
                    if "context" in response and response["context"]:
                        sources = set()
                        for doc in response["context"]:
                            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                                sources.add(doc.metadata['source'])
                        
                        if sources:
                            answer += f"\n\n📚 *Sources: {', '.join(sources)}*"
                    
                except Exception as e:
                    answer = f"❌ Désolé, une erreur s'est produite : {str(e)}"
                
                thinking_placeholder.empty()
                st.markdown(answer)
                
            # Sauvegarde de la réponse
            st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.info("""
        ### 🚀 Pour commencer :
        
        1. **Installez Ollama** : [ollama.ai](https://ollama.ai)
        2. **Téléchargez le modèle** : `ollama run mistral`
        3. **Configurez Google Drive** : 
           - Créez un compte de service Google Cloud
           - Téléchargez le fichier `credentials.json`
           - Remplacez `FOLDER_ID` par l'ID de votre dossier Drive
        4. **Lancez l'application** : `streamlit run app.py`
        """)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #C0C0C0;">
        <p style="font-family: 'Exo 2', sans-serif; font-size: 0.9rem;">
            🏅 Transition Assistant © 2024 | Powered by Local AI
        </p>
    </div>
""", unsafe_allow_html=True)
