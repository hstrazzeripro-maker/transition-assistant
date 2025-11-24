"""
Application Transition Assistant - Version Cloud (corrigée)
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path

# --- IMPORTS ROBUSTES (avec messages clairs si manquant) ---
# Google Drive Loader (package tiers)
try:
    from langchain_google_community import GoogleDriveLoader
except Exception as e:
    GoogleDriveLoader = None
    st.warning("⚠️ Module 'langchain-google-community' non disponible. "
               "Ajoutez 'langchain-google-community' dans requirements.txt.")

# Text splitter (package séparé)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception as e:
    RecursiveCharacterTextSplitter = None
    st.warning("⚠️ Module 'langchain-text-splitters' non disponible. "
               "Ajoutez 'langchain-text-splitters' dans requirements.txt.")

# Hugging Face embeddings / hub via langchain_community
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    HuggingFaceEmbeddings = None
    st.warning("⚠️ Module 'langchain_community.embeddings' non disponible. "
               "Vérifiez 'langchain-community' et 'sentence-transformers' dans requirements.txt.")

try:
    from langchain_community.llms import HuggingFaceHub
except Exception as e:
    HuggingFaceHub = None
    st.warning("⚠️ Module 'langchain_community.llms' non disponible. "
               "Vérifiez 'langchain-community' et 'huggingface_hub' dans requirements.txt.")

# Vectorstore FAISS (via community wrapper)
try:
    from langchain_community.vectorstores import FAISS
except Exception as e:
    FAISS = None
    st.warning("⚠️ Module 'langchain_community.vectorstores' non disponible. "
               "Ajoutez 'langchain-community' et 'faiss-cpu' dans requirements.txt.")

# Chains / prompts (utilisés de façon stable)
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except Exception as e:
    create_stuff_documents_chain = None
    st.warning("⚠️ 'create_stuff_documents_chain' indisponible dans langchain.chains.combine_documents.")

try:
    from langchain_core.prompts import ChatPromptTemplate
except Exception as e:
    ChatPromptTemplate = None
    st.warning("⚠️ 'langchain_core.prompts.ChatPromptTemplate' indisponible. Vérifiez 'langchain-core' dans requirements.txt.")

# --- CONFIGURATION CLOUD / SECRETS ---

IS_CLOUD = 'STREAMLIT_CLOUD' in os.environ or ('google_credentials' in st.secrets)

if IS_CLOUD:
    # Créer credentials.json si présent dans les secrets
    if 'google_credentials' in st.secrets:
        try:
            creds_dict = dict(st.secrets['google_credentials'])
            with open('credentials.json', 'w') as f:
                json.dump(creds_dict, f)
        except Exception as e:
            st.error("❌ Impossible d'écrire credentials.json depuis Streamlit Secrets.")
    FOLDER_ID = st.secrets.get('app_config', {}).get('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = st.secrets.get('app_config', {}).get('HUGGINGFACE_TOKEN', None)
else:
    FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', None)

SERVICE_ACCOUNT_FILE = "credentials.json"

# --- CSS personnalisé (inchangé) ---
def load_css():
    return """
    <style>
    /* (ton CSS ici — inchangé pour la lisibilité) */
    </style>
    """

st.set_page_config(
    page_title="Transition Assistant | Elite Athletes",
    page_icon="🏅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(load_css(), unsafe_allow_html=True)

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
    if HUGGINGFACE_TOKEN and HuggingFaceHub is not None:
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
        # Essayer Ollama en local si disponible
        try:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="mistral", temperature=0.1)
            st.success("✅ Modèle local Ollama connecté")
            return llm
        except Exception:
            st.warning("⚠️ Ni Hugging Face ni Ollama configurés")
            with st.expander("📋 Comment configurer un modèle IA gratuit?"):
                st.markdown("""
                ### Option 1: Hugging Face (Recommandé pour le cloud)
                1. Créez un compte sur huggingface.co
                2. Allez dans Settings > Access Tokens
                3. Créez un token "Read"
                4. Ajoutez-le dans Streamlit Secrets:
                ```toml
                [app_config]
                HUGGINGFACE_TOKEN = "hf_votre_token_ici"
                ```
                ### Option 2: Ollama (Pour usage local)
                1. Installez Ollama
                2. Lancez: `ollama run mistral`
                """)
            return None

# --- INITIALISATION DE LA BASE DE CONNAISSANCES ---

@st.cache_resource
def initialize_knowledge_base():
    """
    Charge les documents du Google Drive et crée l'index de recherche
    """
    # Vérifications préalables
    if GoogleDriveLoader is None:
        st.error("❌ GoogleDriveLoader indisponible. Ajoutez 'langchain-google-community' dans requirements.txt.")
        return None

    if RecursiveCharacterTextSplitter is None:
        st.error("❌ Text splitter indisponible. Ajoutez 'langchain-text-splitters' dans requirements.txt.")
        return None

    if HuggingFaceEmbeddings is None:
        st.error("❌ Embeddings HuggingFace indisponibles. Vérifiez 'langchain-community' et 'sentence-transformers'.")
        return None

    if FAISS is None:
        st.error("❌ FAISS indisponible. Ajoutez 'faiss-cpu' et 'langchain-community' dans requirements.txt.")
        return None

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"⚠️ Fichier '{SERVICE_ACCOUNT_FILE}' introuvable")
        with st.expander("📋 Comment configurer Google Drive?"):
            st.markdown("""
            1. Google Cloud Console: Créez un projet
            2. Activez l'API Drive
            3. Créez un compte de service
            4. Téléchargez le JSON → `credentials.json`
            5. Partagez votre dossier avec l'email du compte
            6. Copiez l'ID du dossier dans les secrets
            """)
        return None

    progress_placeholder = st.empty()
    with progress_placeholder.container():
        with st.spinner("🔄 Chargement de la base de connaissances..."):
            try:
                loader = GoogleDriveLoader(
                    folder_id=FOLDER_ID,
                    file_types=["docx", "doc", "pdf", "txt"],
                    service_account_key=SERVICE_ACCOUNT_FILE,
                    recursive=True
                )
                docs = loader.load()
                if not docs:
                    st.warning("📂 Aucun document trouvé dans le dossier Google Drive.")
                    return None

                # Découpage
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Embeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'}
                )

                # Vectorstore FAISS
                vectorstore = FAISS.from_documents(splits, embeddings)

                progress_placeholder.empty()
                st.success(f"✅ {len(docs)} documents chargés et indexés!")
                return vectorstore

            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
                return None

# --- INTERFACE PRINCIPALE / LOGIQUE CHAT ---

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    llm = get_llm()

    if llm:
        vectorstore = initialize_knowledge_base()

        if vectorstore:
            # Récupérateur
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

            # Template de prompt
            if ChatPromptTemplate is None:
                st.error("❌ ChatPromptTemplate indisponible. Vérifiez 'langchain-core'.")
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                if create_stuff_documents_chain is None:
                    st.error("❌ create_stuff_documents_chain indisponible. Vérifiez votre installation de LangChain.")
                else:
                    # Chaîne qui combine les documents et le LLM
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)

                    # Fonction robuste pour répondre à une question
                    def answer_question(user_input: str) -> str:
                        try:
                            docs = retriever.get_relevant_documents(user_input)
                            # On passe les docs en contexte à la chaîne
                            result = question_answer_chain.invoke({"input": user_input, "context": docs})
                            # result peut être une string ou un dict selon version ; on normalise
                            if isinstance(result, dict):
                                # Chercher une clé 'answer' ou 'output'
                                return result.get("answer") or result.get("output") or str(result)
                            return str(result)
                        except Exception as e:
                            return f"❌ Erreur interne lors de la génération: {str(e)}"

                    # --- Chat UI ---
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
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        with st.chat_message("user"):
                            st.markdown(user_input)

                        with st.chat_message("assistant"):
                            with st.spinner("🤔 Réflexion..."):
                                answer = answer_question(user_input)
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("📂 Configurez Google Drive et les dépendances pour commencer")
    else:
        st.warning("🤖 Configurez un modèle IA (Hugging Face token ou Ollama) pour commencer")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #C0C0C0;">
        <p style="font-family: 'Exo 2', sans-serif; font-size: 0.9rem;">
            🏅 Transition Assistant © 2024 | Cloud Edition
        </p>
    </div>
""", unsafe_allow_html=True)
