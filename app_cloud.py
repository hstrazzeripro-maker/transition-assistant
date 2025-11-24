"""
Application Transition Assistant - Version Cloud (corrigée, sans langchain.chains)
"""

import streamlit as st
import os
import json
import time
from datetime import datetime
from pathlib import Path

# --- IMPORTS STABLES ---
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION CLOUD / SECRETS ---
IS_CLOUD = 'STREAMLIT_CLOUD' in os.environ or ('google_credentials' in st.secrets)

if IS_CLOUD:
    if 'google_credentials' in st.secrets:
        creds_dict = dict(st.secrets['google_credentials'])
        with open('credentials.json', 'w') as f:
            json.dump(creds_dict, f)
    FOLDER_ID = st.secrets.get('app_config', {}).get('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = st.secrets.get('app_config', {}).get('HUGGINGFACE_TOKEN', None)
else:
    FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', None)

SERVICE_ACCOUNT_FILE = "credentials.json"

# --- CONFIGURATION DU MODÈLE ---
@st.cache_resource
def get_llm():
    if HUGGINGFACE_TOKEN:
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
            return None
    else:
        try:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="mistral", temperature=0.1)
            st.success("✅ Modèle local Ollama connecté")
            return llm
        except Exception:
            st.warning("⚠️ Ni Hugging Face ni Ollama configurés")
            return None

# --- INITIALISATION DE LA BASE DE CONNAISSANCES ---
@st.cache_resource
def initialize_knowledge_base():
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error(f"⚠️ Fichier '{SERVICE_ACCOUNT_FILE}' introuvable")
        return None

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
                st.warning("📂 Aucun document trouvé")
                return None

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"✅ {len(docs)} documents chargés et indexés!")
            return vectorstore

        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            return None

# --- INTERFACE PRINCIPALE ---
llm = get_llm()
if llm:
    vectorstore = initialize_knowledge_base()
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

        def answer_question(user_input: str) -> str:
            docs = retriever.get_relevant_documents(user_input)
            result = question_answer_chain.invoke({"input": user_input, "context": docs})
            return str(result)

        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome = "🌟 Bienvenue / Welcome! Posez vos questions."
            st.session_state.messages.append({"role": "assistant", "content": welcome})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

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
        st.info("📂 Configurez Google Drive pour commencer")
else:
    st.warning("🤖 Configurez un modèle IA pour commencer")
