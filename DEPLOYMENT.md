# 🚀 Guide de Déploiement - Transition Assistant

## 📊 Comparaison des Solutions Gratuites

| Plateforme | Difficulté | Temps | Limitations | Idéal pour |
|------------|------------|-------|-------------|------------|
| **Streamlit Cloud** | ⭐ Facile | 5 min | 1 app gratuite | Démo rapide |
| **Render** | ⭐⭐ Moyen | 15 min | 750h/mois | Production |
| **Railway** | ⭐⭐ Moyen | 10 min | $5 crédit/mois | Tests |
| **Ngrok** | ⭐ Facile | 2 min | Temporaire | Partage rapide |

---

## 🔥 Option 1: Streamlit Cloud (RECOMMANDÉ)

### ✅ Avantages
- 100% gratuit pour 1 app publique
- Déploiement en 1 clic
- HTTPS automatique
- Mise à jour automatique via GitHub

### 📋 Étapes

#### 1. Préparer votre code sur GitHub

```bash
# Créer un nouveau repo GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/VOTRE_USERNAME/transition-assistant.git
git push -u origin main
```

#### 2. Configurer Streamlit Cloud

1. Allez sur https://streamlit.io/cloud
2. Connectez-vous avec GitHub
3. Cliquez "New app"
4. Sélectionnez votre repo
5. Branch: `main`
6. Main file path: `app.py`

#### 3. Ajouter les Secrets

Dans Streamlit Cloud, allez dans Settings > Secrets et ajoutez:

```toml
# Contenu de credentials.json
[google_credentials]
type = "service_account"
project_id = "votre-project-id"
private_key_id = "votre-key-id"
private_key = """-----BEGIN PRIVATE KEY-----
VOTRE_PRIVATE_KEY
-----END PRIVATE KEY-----"""
client_email = "votre-email@projet.iam.gserviceaccount.com"
client_id = "votre-client-id"

# Config
[app_config]
GOOGLE_DRIVE_FOLDER_ID = "votre-folder-id"
OLLAMA_MODEL = "mistral"
```

#### 4. Modifier app.py pour les secrets

```python
import streamlit as st
import json

# Pour Streamlit Cloud
if 'google_credentials' in st.secrets:
    # Créer credentials.json depuis secrets
    creds = dict(st.secrets['google_credentials'])
    with open('credentials.json', 'w') as f:
        json.dump(creds, f)
    
    # Utiliser config depuis secrets
    FOLDER_ID = st.secrets['app_config']['GOOGLE_DRIVE_FOLDER_ID']
else:
    # Local development
    from config import GOOGLE_DRIVE_FOLDER_ID as FOLDER_ID
```

#### 5. URL Finale

Votre app sera disponible sur:
```
https://transition-assistant-XXXXX.streamlit.app
```

---

## 🐳 Option 2: Render (Plus Stable)

### ✅ Avantages
- 750 heures gratuites/mois
- Support Docker
- Bases de données gratuites
- Domaine personnalisé possible

### 📋 Étapes

#### 1. Créer render.yaml

```yaml
services:
  - type: web
    name: transition-assistant
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: STREAMLIT_SERVER_PORT
        value: 8501
      - key: STREAMLIT_SERVER_ADDRESS
        value: 0.0.0.0
    autoDeploy: true
```

#### 2. Déployer sur Render

1. Créez un compte sur https://render.com
2. New > Web Service
3. Connectez votre repo GitHub
4. Choisissez "Docker" comme environnement
5. Cliquez "Create Web Service"

#### 3. Variables d'environnement

Dans Render Dashboard > Environment:
- Ajoutez vos secrets Google
- Ajoutez FOLDER_ID

#### 4. URL Finale

```
https://transition-assistant.onrender.com
```

---

## 🚂 Option 3: Railway

### ✅ Avantages
- Déploiement ultra-rapide
- $5 de crédit gratuit/mois
- Support Ollama natif
- Métriques en temps réel

### 📋 Étapes

#### 1. Installer Railway CLI

```bash
npm install -g @railway/cli
railway login
```

#### 2. Créer railway.json

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE"
  },
  "deploy": {
    "numReplicas": 1,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

#### 3. Déployer

```bash
railway up
railway domain
```

---

## 🌍 Option 4: Ngrok (Test Rapide)

### ✅ Avantages
- Partage instantané
- Aucune configuration
- Parfait pour les démos

### 📋 Étapes

#### 1. Installer Ngrok

```bash
# Windows (Chocolatey)
choco install ngrok

# Mac (Homebrew)
brew install ngrok

# Linux
snap install ngrok
```

#### 2. Lancer votre app localement

```bash
streamlit run app.py
```

#### 3. Créer le tunnel

```bash
ngrok http 8501
```

#### 4. URL temporaire

```
https://abc123.ngrok.io
```

---

## ⚠️ IMPORTANT: Adapter pour le Cloud

### Problème avec Ollama

Ollama ne fonctionne pas sur les plateformes cloud gratuites. Voici les alternatives:

### Solution 1: API Hugging Face (GRATUIT)

```python
# Remplacer Ollama par Hugging Face
from langchain_community.llms import HuggingFaceHub

# Configuration
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_VOTRE_TOKEN"

# Utiliser un modèle gratuit
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 512}
)
```

Pour obtenir un token gratuit:
1. Créez un compte sur https://huggingface.co
2. Settings > Access Tokens
3. New Token > Read

### Solution 2: Groq API (GRATUIT)

```python
# Installation
pip install groq

# Configuration
from groq import Groq

client = Groq(api_key="gsk_VOTRE_CLE_GROQ")

# Utilisation
response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="mixtral-8x7b-32768",  # Gratuit!
    temperature=0.1
)
```

Pour obtenir une clé Groq:
1. Allez sur https://console.groq.com
2. Créez un compte gratuit
3. API Keys > Create

### Solution 3: Together AI (GRATUIT avec limites)

```python
# Installation
pip install together

# Configuration  
import together

together.api_key = "VOTRE_CLE_TOGETHER"

# Utilisation
response = together.Complete.create(
    prompt=prompt,
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens=512,
    temperature=0.1
)
```

---

## 📝 Fichier app_cloud.py Adapté

```python
import streamlit as st
import os
from langchain_huggingface import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
# ... autres imports

# Configuration pour le cloud
if 'HUGGINGFACE_TOKEN' in os.environ:
    # Version cloud
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=os.environ['HUGGINGFACE_TOKEN'],
        model_kwargs={"temperature": 0.1}
    )
else:
    # Version locale avec Ollama
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="mistral")

# Le reste du code reste identique
```

---

## 🔐 Sécurité pour la Production

### 1. Variables d'Environnement

Jamais de secrets dans le code! Utilisez:

```python
import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
API_KEY = os.getenv("API_KEY")
```

### 2. Fichier .env.example

```env
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
HUGGINGFACE_TOKEN=your_token_here
GROQ_API_KEY=your_key_here
```

### 3. Authentication Utilisateurs

Pour Streamlit Cloud:

```python
import streamlit as st
import hmac

def check_password():
    """Returns True if password is correct."""
    def password_entered():
        """Checks whether password is correct."""
        if hmac.compare_digest(st.session_state["password"], 
                               st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, 
                     key="password")
        return False
    
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, 
                     key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Votre app ici
    st.write("Welcome!")
```

---

## 🚀 Commandes de Déploiement Rapide

### GitHub + Streamlit Cloud (5 min)

```bash
# 1. Initialiser Git
git init
git add .
git commit -m "Deploy to Streamlit Cloud"

# 2. Créer repo GitHub
gh repo create transition-assistant --public --push

# 3. Aller sur streamlit.io/cloud
# 4. Connecter et déployer
```

### Docker + Render (10 min)

```bash
# 1. Build Docker
docker build -t transition-assistant .

# 2. Push vers GitHub
git push origin main

# 3. Déployer sur Render
# Via interface web
```

### Railway (3 min)

```bash
# 1. Login
railway login

# 2. Init projet
railway init

# 3. Deploy
railway up

# 4. Get URL
railway domain
```

---

## 📊 Monitoring et Analytics

### Ajouter Google Analytics

```python
# Dans app.py
GA_TRACKING_CODE = """
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>
"""

st.markdown(GA_TRACKING_CODE, unsafe_allow_html=True)
```

---

## 🆘 Dépannage Déploiement

| Problème | Solution |
|----------|----------|
| "Module not found" | Vérifiez requirements.txt |
| "Port binding failed" | Utilisez $PORT ou 8501 |
| "Memory limit" | Réduisez le modèle/cache |
| "Timeout" | Ajoutez health checks |
| "Credentials error" | Vérifiez les secrets/env vars |

---

## 📱 Tester sur Mobile

1. **Tunnel local**: `ngrok http 8501`
2. **QR Code**: Générez un QR vers votre URL
3. **Responsive**: Testez sur différentes tailles

---

## 🎯 Checklist Finale

- [ ] Code sur GitHub
- [ ] Secrets configurés
- [ ] Requirements.txt à jour
- [ ] Dockerfile testé
- [ ] Variables d'environnement
- [ ] HTTPS activé
- [ ] Domaine personnalisé (optionnel)
- [ ] Monitoring configuré
- [ ] Backup des données

---

## 🌟 URL Finale

Votre application sera accessible à:

```
https://votre-app.streamlit.app
https://votre-app.onrender.com
https://votre-app.railway.app
```

Partagez le lien avec vos athlètes! 🏅
