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
from langchain.schema import Document

# Imports pour le diagnostic Drive
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
    google_creds = creds_dict if creds_dict else None
    wrote_temp_file = False

    if google_creds:
        try:
            with open(SERVICE_ACCOUNT_FILE, "w") as f:
                json.dump(google_creds, f, indent=2)
            wrote_temp_file = True
        except Exception as e:
            st.error(f"Impossible d'écrire {SERVICE_ACCOUNT_FILE}: {e}")
            return None
    else:
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            st.error("⚠️ Aucun credentials disponible.")
            return None

    try:
        with st.spinner("🔎 Chargement Google Drive..."):
            # Essai standard avec GoogleDriveLoader
            try:
                loader = GoogleDriveLoader(
                    folder_id=FOLDER_ID,
                    file_types=["document", "pdf"],
                    service_account_key=SERVICE_ACCOUNT_FILE,
                    recursive=True
                )
                docs = loader.load()
            except Exception as e:
                st.warning(f"GoogleDriveLoader a levé une exception: {e}")
                docs = []

            if not docs:
                # Fallback manuel via API Drive
                st.info("Fallback : téléchargement manuel via Google Drive API...")
                SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
                creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
                service = build("drive", "v3", credentials=creds)

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
                            pdf_stream = BytesIO(data)
                            reader = PdfReader(pdf_stream)
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
        try:
            if wrote_temp_file and os.path.exists(SERVICE_ACCOUNT_FILE):
                os.remove(SERVICE_ACCOUNT_FILE)
        except Exception:
            pass

# --- DIAGNOSTIC GOOGLE DRIVE (UI) ---
def drive_diagnostic_ui():
    st.sidebar.header("Diagnostic Google Drive")
    if st.sidebar.button("Run Drive diagnostic"):
        st.subheader("Diagnostic secrets et test Google Drive")
        app_conf = st.secrets.get("app_config", {})
        google_creds_raw = st.secrets.get("google_credentials", {})

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
                st.table([{"
