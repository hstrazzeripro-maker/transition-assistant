"""
app_cloud.py

Transition Assistant - Cloud version
- Google Drive ingestion (with fallback)
- PDF text extraction via pypdf
- Lang detection (langdetect) to reply in same language
- Response length control (slider: court / moyen / approfondi)
- Modern sporty UI theme with bronze / silver / gold accents
- Compatible with LangChain 0.2+ (retriever.invoke)
"""

import os
import json
from io import BytesIO
from typing import Optional

import streamlit as st

# LangChain / embeddings / vectorstore
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build

# PDF extraction
from pypdf import PdfReader

# Language detection
from langdetect import detect, LangDetectException

# --- Config / secrets ---
SERVICE_ACCOUNT_FILE = "credentials.json"
IS_CLOUD = 'STREAMLIT_CLOUD' in os.environ or ('google_credentials' in st.secrets)

if IS_CLOUD:
    creds_raw = st.secrets.get('google_credentials', {})
    try:
        CREDS_DICT = json.loads(json.dumps(creds_raw)) if creds_raw else {}
    except Exception:
        CREDS_DICT = dict(creds_raw) if creds_raw else {}
    FOLDER_ID = st.secrets.get('app_config', {}).get('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = st.secrets.get('app_config', {}).get('HUGGINGFACE_TOKEN', None)
else:
    CREDS_DICT = {}
    FOLDER_ID = os.getenv('GOOGLE_DRIVE_FOLDER_ID', 'VOTRE_FOLDER_ID')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN', None)

# --- Styling (bronze / silver / gold sporty theme) ---
st.set_page_config(page_title="Transition Assistant | Elite Athletes", page_icon="🏅", layout="wide")

_THEME_CSS = """
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0f1724 0%, #071022 40%, #071018 100%);
  color: #f7f7f7;
}

/* Card / main container */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}

/* Header */
.header {
  display:flex;
  align-items:center;
  gap:12px;
  margin-bottom: 12px;
}
.brand {
  font-weight:700;
  font-size:20px;
  color: #ffd27f; /* gold */
  letter-spacing: 0.6px;
}
.subtitle {
  color: #d6d6d6;
  font-size:12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #071018 0%, #08121a 100%);
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Buttons and inputs */
.stButton>button {
  background: linear-gradient(90deg, #b08a4b, #ffd27f);
  color: #071018;
  font-weight:600;
  border-radius:10px;
  padding: 8px 12px;
  border: none;
}
.stButton>button:hover {
  filter: brightness(1.05);
}

/* Chat bubbles */
.streamlit-expanderHeader {
  color: #f0f0f0;
}

/* Message styling */
.chat-user {
  background: linear-gradient(90deg, #c0c0c0, #e6e6e6);
  color: #071018;
  padding: 10px;
  border-radius: 10px;
}
.chat-assistant {
  background: linear-gradient(90deg, #6b4a2a, #b08a4b); /* bronze/gold */
  color: #fff;
  padding: 10px;
  border-radius: 10px;
}

/* Slider label */
.slider-label {
  color: #f0f0f0;
  font-weight:600;
}

/* Small muted text */
.small-muted {
  color: #9aa4b2;
  font-size:12px;
}
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)

# --- Utilities ---
def mask(s: Optional[str], keep_start: int = 6, keep_end: int = 6) -> Optional[str]:
    if not s:
        return None
    if len(s) <= keep_start + keep_end:
        return "*****"
    return s[:keep_start] + "..." + s[-keep_end:]

# --- LLM setup ---
@st.cache_resource
def get_llm():
    if HUGGINGFACE_TOKEN:
        try:
            # Use conversational task to match provider expectations
            endpoint = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                huggingfacehub_api_token=HUGGINGFACE_TOKEN,
                task="conversational",
                temperature=0.1,
                max_new_tokens=512,
                top_p=0.95,
                repetition_penalty=1.1
            )
            llm = ChatHuggingFace(llm=endpoint)
            return llm
        except Exception:
            return None
    else:
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="mistral", temperature=0.1)
        except Exception:
            return None

# --- Knowledge base initialization (GoogleDriveLoader with forced credentials.json and fallback) ---
@st.cache_resource
def initialize_knowledge_base():
    wrote_temp_file = False
    # Force write credentials.json if provided in st.secrets
    if CREDS_DICT:
        try:
            with open(SERVICE_ACCOUNT_FILE, "w", encoding="utf-8") as f:
                json.dump(CREDS_DICT, f, ensure_ascii=False, indent=2)
            wrote_temp_file = True
        except Exception as e:
            st.error(f"Impossible d'écrire {SERVICE_ACCOUNT_FILE}: {e}")
            return None
    elif not os.path.exists(SERVICE_ACCOUNT_FILE):
        st.error("⚠️ Aucun credentials disponible (st.secrets absent et credentials.json introuvable).")
        return None

    try:
        with st.spinner("🔎 Chargement des documents depuis Google Drive..."):
            docs = []
            # Try GoogleDriveLoader (expects a path)
            try:
                loader = GoogleDriveLoader(
                    folder_id=FOLDER_ID,
                    file_types=["document", "pdf"],
                    service_account_key=str(SERVICE_ACCOUNT_FILE),
                    recursive=True
                )
                docs = loader.load()
            except Exception as e:
                st.warning(f"GoogleDriveLoader (service_account_key) a échoué: {e}")

            # If loader returned nothing, fallback to manual Drive API + pypdf
            if not docs:
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

            st.write(f"📂 Documents trouvés pour indexation: {len(docs)}")
            if not docs:
                st.warning("📂 Aucun document exploitable trouvé.")
                return None

            # Split and index
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = FAISS.from_documents(splits, embeddings)
            return vectorstore

    except Exception as e:
        st.error(f"❌ Erreur lors de l'initialisation: {e}")
        return None

    finally:
        # cleanup credentials.json if we wrote it from secrets
        try:
            if wrote_temp_file and os.path.exists(SERVICE_ACCOUNT_FILE):
                os.remove(SERVICE_ACCOUNT_FILE)
        except Exception:
            pass

# --- Diagnostic UI (sidebar) ---
def drive_diagnostic_ui():
    st.sidebar.header("Diagnostic Google Drive")
    st.sidebar.write("Vérifie les secrets et l'accès du compte de service au dossier Drive.")
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
                st.table([{"id": f["id"], "name": f["name"], "mimeType": f["mimeType"]} for f in files])
            else:
                st.warning("Aucun fichier trouvé. Vérifie : Folder ID, partage du dossier avec client_email, types de fichiers (Google Docs / PDF).")

        except Exception as e:
            st.error(f"Erreur lors du test Drive: {e}")

# --- Main UI layout ---
st.markdown('<div class="header"><div class="brand">Transition Assistant</div><div class="subtitle">Elite Athletes • Career Transition Support</div></div>', unsafe_allow_html=True)

drive_diagnostic_ui()

llm = get_llm()
if not llm:
    st.warning("⚠️ Aucun modèle IA configuré. Vérifie HUGGINGFACE_TOKEN ou la configuration locale.")
else:
    vectorstore = initialize_knowledge_base()
    if not vectorstore:
        st.info("📂 Configurez Google Drive pour commencer (ou vérifiez le diagnostic).")
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # System prompt template with placeholders for language instruction and context
        system_prompt_template = """
You are an expert assistant for elite athlete career transition.

INSTRUCTIONS (follow exactly):
- Detect the user's input language and respond in the SAME language.
- Provide a LONG, well-structured answer unless the user requests short: include a short introduction, 3–6 key points with brief explanations, at least one concrete example, 3 practical action steps, and a concise conclusion.
- Be professional, empathetic, and practical.
- Use the following language instruction: {language_instruction}
- Use ONLY the provided context when it is relevant; if context is insufficient, say so and provide general best-practice guidance.

Context:
{context}

User question:
{input}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_template),
            ("human", "{input}"),
        ])

        chain = prompt | llm

        # Response length control UI (slider)
        st.sidebar.markdown("### Réglage de la longueur de la réponse")
        length_choice = st.sidebar.select_slider(
            label="Niveau de détail",
            options=["Court", "Moyen", "Approfondi"],
            value="Moyen"
        )
        st.sidebar.markdown('<div class="small-muted">Court = 1 paragraphe; Moyen = 3-4 paragraphes; Approfondi = 5+ paragraphes détaillés</div>', unsafe_allow_html=True)

        # Chat session state
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "🌟 Bienvenue ! Posez votre question en français, anglais, italien ou autre — je répondrai dans la même langue."}]

        # Display chat history
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            if role == "user":
                st.markdown(f'<div class="chat-user">{content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">{content}</div>', unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("💬 Votre question / Your question...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.markdown(f'<div class="chat-user">{user_input}</div>', unsafe_allow_html=True)

            # Build language instruction based on detection and length choice
            try:
                detected_lang = detect(user_input)
            except LangDetectException:
                detected_lang = "fr"

            lang_map_instruction = {
                "fr": "Réponds en français.",
                "en": "Respond in English.",
                "it": "Rispondi in italiano.",
                "es": "Responde en español.",
                "de": "Antworte auf Deutsch."
            }
            language_instruction = lang_map_instruction.get(detected_lang, f"Respond in the user's language ({detected_lang}) if possible.")

            # Adjust verbosity hint
            verbosity_hint = {
                "Court": {
                    "fr": "Donne une réponse courte (1 paragraphe, 2-4 phrases).",
                    "en": "Give a short answer (1 paragraph, 2-4 sentences)."
                },
                "Moyen": {
                    "fr": "Donne une réponse de longueur moyenne (3-4 paragraphes, points clés et actions).",
                    "en": "Give a medium-length answer (3-4 paragraphs, key points and actions)."
                },
                "Approfondi": {
                    "fr": "Donne une réponse approfondie (5+ paragraphes, exemples concrets, étapes détaillées).",
                    "en": "Give an in-depth answer (5+ paragraphs, concrete examples, detailed steps)."
                }
            }
            # Choose hint in detected language if available, else English
            verbosity_text = verbosity_hint.get(length_choice, {}).get(detected_lang) or verbosity_hint.get(length_choice, {}).get("en")

            # Retrieve docs via retriever.invoke (LC 0.2+)
            try:
                docs = retriever.invoke(user_input)  # List[Document]
            except Exception:
                docs = []

            context = "\n\n".join([d.page_content or "" for d in docs]) if docs else ""

            # Compose inputs for chain.invoke
            # We pass language_instruction and context; the ChatPromptTemplate will include them
            # Also append verbosity_text into the user input to nudge length
            augmented_input = f"{verbosity_text}\n\n{user_input}"

            with st.spinner("🤔 Réflexion..."):
                try:
                    result = chain.invoke({
                        "language_instruction": language_instruction,
                        "context": context,
                        "input": augmented_input
                    })
                    # result may be an object; extract textual content
                    answer_text = getattr(result, "content", None) or str(result)
                except Exception as e:
                    answer_text = f"Erreur lors de la génération de la réponse: {e}"

            # Append and display assistant message
            st.markdown(f'<div class="chat-assistant">{answer_text}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
