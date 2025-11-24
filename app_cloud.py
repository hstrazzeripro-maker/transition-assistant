"""
Application Transition Assistant - Version Cloud
Inclut diagnostic Google Drive et extraction texte des PDF avec pypdf
"""

import streamlit as st
import os
import json
from typing import Optional
from io import BytesIO

from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document  # ✅ import corrigé

# Imports Drive + PDF
from google.oauth2 import service_account
from googleapiclient.discovery import build
from pypdf import PdfReader

# --- CONFIGURATION CLOUD / SECRETS ---
IS_CLOUD = 'STREAMLIT_CLOUD' in os.environ or ('google_credentials' in st.secrets)

if IS_CLOUD:
    creds_raw = st.secrets.get('google_credentials', {})
    try:
        creds_dict = json.loads(json.dumps(creds_raw)) if creds_raw else {}
    except Exception:
        creds_dict = dict(creds_raw) if creds_raw else {}
    FOLDER_ID = st.secrets.get('app_config', {}).get('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = st.secrets.get('app_config', {}).get('HUGGINGFACE_TOKEN', None)
else:
    creds_dict = {}
    FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', None)

SERVICE_ACCOUNT_FILE = "credentials.json"

# --- UTILITAIRES ---
def mask(s: Optional[str], keep_start: int = 6, keep_end: int = 6) -> Optional[str]:
    if not s:
        return None
    if len(s) <= keep_start + keep_end:
        return "*****"
    return s[:keep_start] + "..." + s[-keep_end:]

# --- CONFIGURATION DU MODÈLE ---
@st.cache_resource
def get_llm():
    if HUGGINGFACE_TOKEN:
        try:
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                task="text-generation",
                temperature=0.1,
                max_new_tokens=512,
                top_p=0.95,
                repetition_penalty=1.1
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
    # 1) Forcer l’écriture locale du credentials.json depuis st.secrets si dispo
    google_creds_raw = st.secrets.get("google_credentials", {})
    try:
        google_creds = json.loads(json.dumps(google_creds_raw)) if google_creds_raw else {}
    except Exception:
        google_creds = dict(google_creds_raw) if google_creds_raw else {}

    wrote_temp_file = False
    if google_creds:
        try:
            with open(SERVICE_ACCOUNT_FILE, "w", encoding="utf-8") as f:
                json.dump(google_creds, f, ensure_ascii=False, indent=2)
            wrote_temp_file = True
        except Exception as e:
            st.error(f"Impossible d'écrire {SERVICE_ACCOUNT_FILE}: {e}")
            return None

    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error("⚠️ credentials.json introuvable et st.secrets['google_credentials'] absent.")
        return None

    try:
        with st.spinner("🔎 Chargement Google Drive..."):
            # Essai A — passer un chemin (str) via service_account_key
            docs = []
            try:
                loader = GoogleDriveLoader(
                    folder_id=FOLDER_ID,
                    file_types=["document", "pdf"],
                    service_account_key=str(SERVICE_ACCOUNT_FILE),  # chemin attendu par ta version
                    recursive=True
                )
                docs = loader.load()
            except Exception as e1:
                st.warning(f"GoogleDriveLoader (service_account_key) a échoué: {e1}")

            # Fallback manuel si le loader ne retourne rien
            if not docs:
                st.info("Fallback : téléchargement manuel via Google Drive API...")
                SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
                creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
                service = build("drive", "v3", credentials=creds)

                # Lister les fichiers du dossier
                resp = service.files().list(
                    q=f"'{FOLDER_ID}' in parents and trashed = false",
                    fields="files(id,name,mimeType)",
                    pageSize=500,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True
                ).execute()
                files = resp.get("files", [])

                docs = []
                for f in files:
                    fid = f["id"]
                    name = f.get("name", "")
                    mime = f.get("mimeType", "")
                    try:
                        if mime == "application/vnd.google-apps.document":
                            exported = service.files().export(fileId=fid, mimeType="text/plain").execute()
                            text = exported.decode("utf-8") if isinstance(exported, bytes) else str(exported)
                        elif mime == "application/pdf":
                            data = service.files().get_media(fileId=fid).execute()
                            reader = PdfReader(BytesIO(data))
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() or ""
                            if not text.strip():
                                text = f"[PDF sans texte exploitable] {name}"
                        else:
                            try:
                                exported = service.files().export(fileId=fid, mimeType="text/plain").execute()
                                text = exported.decode("utf-8") if isinstance(exported, bytes) else str(exported)
                            except Exception:
                                text = f"[Type non supporté: {mime}] {name}"
                        docs.append(Document(page_content=text, metadata={"id": fid, "name": name, "mimeType": mime}))
                    except Exception as e:
                        st.warning(f"Erreur téléchargement fichier {name} ({fid}): {e}")

            st.write(f"📂 Nombre de documents prêts à être indexés: {len(docs)}")
            if not docs:
                st.warning("📂 Aucun document exploitable trouvé.")
                return None

            # Découpage + indexation
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"✅ {len(docs)} documents chargés et indexés!")
            return vectorstore

    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation: {str(e)}")
        return None

    finally:
        # Nettoyage : supprimer credentials.json s’il vient de st.secrets (optionnel)
        try:
            if wrote_temp_file and os.path.exists(SERVICE_ACCOUNT_FILE):
                os.remove(SERVICE_ACCOUNT_FILE)
        except Exception:
            pass

# --- DIAGNOSTIC GOOGLE DRIVE (UI) ---
def drive_diagnostic_ui():
    st.sidebar.header("Diagnostic Google Drive")
    st.sidebar.write("Vérifie les secrets et l'accès du compte de service au dossier Drive.")
    if st.sidebar.button("Run Drive diagnostic"):
        st.subheader("Diagnostic secrets et test Google Drive")
        app_conf = st.secrets.get("app_config", {})
        google_creds_raw = st.secrets.get("google_credentials", {})

        # Conversion sûre AttrDict -> dict
        try:
            google_creds = json.loads(json.dumps(google_creds_raw)) if google_creds_raw else {}
        except Exception:
            google_creds = dict(google_creds_raw) if google_creds_raw else {}

        st.write("app_config keys:", list(app_conf.keys()))
        st.write("google_credentials keys:", list(google_creds.keys()))
        st.write("GOOGLE_DRIVE_FOLDER_ID:", mask(app_conf.get("GOOGLE_DRIVE_FOLDER_ID", "")))
        st.write("client_email:", mask(google_creds.get("client_email", "")))

        if not google_creds:
            st.error("Aucun google_credentials trouvé.")
            return

        FOLDER_ID_LOCAL = app_conf.get("GOOGLE_DRIVE_FOLDER_ID", "")
        if not FOLDER_ID_LOCAL:
            st.warning("Aucun FOLDER_ID trouvé.")
            return

        SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
        try:
            creds = service_account.Credentials.from_service_account_info(google_creds, scopes=SCOPES)
            service = build("drive", "v3", credentials=creds)

            resp = service.files().list(
                q=f"'{FOLDER_ID_LOCAL}' in parents and trashed = false",
                fields="files(id,name,mimeType)",
                pageSize=100,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True
            ).execute()

            files = resp.get("files", [])
            st.write("Nombre de fichiers trouvés:", len(files))
            if files:
                st.table([{"id": f["id"], "name": f["name"], "mimeType": f["mimeType"]} for f in files])
            else:
                st.warning("Aucun fichier trouvé. Vérifie : Folder ID, partage du dossier avec client_email, types de fichiers (Google Docs / PDF).")

        except Exception as e:
            st.error(f"Erreur lors du test Drive: {e}")

# --- INTERFACE PRINCIPALE ---
st.set_page_config(page_title="Transition Assistant | Elite Athletes", page_icon="🏅", layout="wide")

# Affiche le diagnostic dans la sidebar (bouton pour lancer)
drive_diagnostic_ui()

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

        Context:
        {context}

        Question:
        {input}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        chain = prompt | llm

        def answer_question(user_input: str) -> str:
            docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([d.page_content or "" for d in docs]) if docs else "No context available."
            result = chain.invoke({"input": user_input, "context": context})
            return getattr(result, "content", str(result))

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "🌟 Bienvenue / Welcome! Posez vos questions."}]

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
