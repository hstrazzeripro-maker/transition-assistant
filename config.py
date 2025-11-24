"""
Configuration file for Transition Assistant
Fichier de configuration pour l'Assistant Transition
"""

import os
from pathlib import Path

# === GOOGLE DRIVE CONFIGURATION ===
# Remplacez par l'ID de votre dossier Google Drive "Transition"
# Pour trouver l'ID : 
# 1. Ouvrez le dossier dans Google Drive
# 2. L'URL sera comme : drive.google.com/drive/folders/XXXXXXXXXXXXXXX
# 3. Copiez la partie XXXXXXXXXXXXXXX

GOOGLE_DRIVE_FOLDER_ID = "VOTRE_FOLDER_ID_ICI"

# Chemin vers votre fichier de credentials Google
# Pour créer ce fichier :
# 1. Allez sur Google Cloud Console (console.cloud.google.com)
# 2. Créez un nouveau projet ou sélectionnez-en un existant
# 3. Activez l'API Google Drive
# 4. Créez un compte de service (Service Account)
# 5. Créez une clé JSON et téléchargez-la
# 6. Renommez le fichier en "credentials.json" et placez-le à la racine du projet

GOOGLE_SERVICE_ACCOUNT_FILE = "credentials.json"

# === MODÈLE IA LOCAL (OLLAMA) ===
# Modèles disponibles : mistral, llama2, codellama, vicuna
# Pour installer un modèle : ollama run <nom_du_modele>

OLLAMA_MODEL = "mistral"  # Recommandé pour le multilinguisme
OLLAMA_TEMPERATURE = 0.1  # 0 = déterministe, 1 = créatif
OLLAMA_MAX_TOKENS = 2048  # Longueur maximale de la réponse

# === EMBEDDINGS (GRATUITS) ===
# Modèles HuggingFace pour la vectorisation
# Options recommandées :
# - "sentence-transformers/all-MiniLM-L6-v2" (Rapide, anglais)
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (Multilingue)
# - "sentence-transformers/all-mpnet-base-v2" (Plus précis, anglais)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === PARAMÈTRES DE RECHERCHE ===
# Nombre de documents pertinents à récupérer
RETRIEVER_K = 3  # Augmentez pour plus de contexte (mais réponses plus lentes)

# Taille des chunks de texte
CHUNK_SIZE = 1000  # Caractères par chunk
CHUNK_OVERLAP = 200  # Chevauchement entre chunks

# === INTERFACE ===
# Thème de couleurs
THEME_COLORS = {
    "gold": "#FFD700",
    "silver": "#C0C0C0", 
    "bronze": "#CD7F32",
    "background": "#1a1a2e",
    "glass": "rgba(255, 255, 255, 0.1)"
}

# Messages système
WELCOME_MESSAGE = {
    "fr": """
    🌟 **Bienvenue sur l'Assistant Transition !**
    
    Je suis votre expert en reconversion professionnelle pour athlètes de haut niveau.
    Posez-moi vos questions sur :
    - La planification de carrière
    - Les compétences transférables
    - Les opportunités professionnelles
    - Le développement personnel
    """,
    "en": """
    🌟 **Welcome to Transition Assistant!**
    
    I'm your expert in career transition for elite athletes.
    Ask me about:
    - Career planning
    - Transferable skills
    - Professional opportunities
    - Personal development
    """
}

ERROR_MESSAGES = {
    "no_ollama": {
        "fr": "❌ Ollama n'est pas installé ou n'est pas lancé. Veuillez suivre les instructions d'installation.",
        "en": "❌ Ollama is not installed or not running. Please follow the installation instructions."
    },
    "no_drive": {
        "fr": "❌ Impossible de se connecter à Google Drive. Vérifiez vos credentials.",
        "en": "❌ Unable to connect to Google Drive. Please check your credentials."
    },
    "no_docs": {
        "fr": "📂 Aucun document trouvé dans le dossier spécifié.",
        "en": "📂 No documents found in the specified folder."
    }
}

# === CHEMINS ===
# Dossier pour le cache local
CACHE_DIR = Path.home() / ".transition_assistant_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Dossier pour les logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# === OPTIONS AVANCÉES ===
# Active le mode debug (plus de logs)
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Utilise GPU si disponible (pour les embeddings)
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

# Limite de mémoire pour FAISS (en MB)
FAISS_MEMORY_LIMIT = 1024  # 1GB

# Timeout pour les requêtes (en secondes)
REQUEST_TIMEOUT = 30

# === VALIDATION ===
def validate_config():
    """Vérifie que la configuration est valide"""
    errors = []
    
    if GOOGLE_DRIVE_FOLDER_ID == "VOTRE_FOLDER_ID_ICI":
        errors.append("❌ Veuillez configurer GOOGLE_DRIVE_FOLDER_ID dans config.py")
    
    if not Path(GOOGLE_SERVICE_ACCOUNT_FILE).exists():
        errors.append(f"❌ Fichier '{GOOGLE_SERVICE_ACCOUNT_FILE}' introuvable")
    
    return errors

# === EXPORT ===
__all__ = [
    'GOOGLE_DRIVE_FOLDER_ID',
    'GOOGLE_SERVICE_ACCOUNT_FILE',
    'OLLAMA_MODEL',
    'OLLAMA_TEMPERATURE',
    'OLLAMA_MAX_TOKENS',
    'EMBEDDING_MODEL',
    'RETRIEVER_K',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'THEME_COLORS',
    'WELCOME_MESSAGE',
    'ERROR_MESSAGES',
    'CACHE_DIR',
    'LOG_DIR',
    'DEBUG_MODE',
    'USE_GPU',
    'validate_config'
]
