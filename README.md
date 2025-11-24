# 🏅 Transition Assistant - Chatbot IA pour Athlètes

<div align="center">
  
![Version](https://img.shields.io/badge/version-1.0.0-gold.svg)
![License](https://img.shields.io/badge/license-MIT-silver.svg)
![Python](https://img.shields.io/badge/python-3.8+-bronze.svg)

**Assistant IA gratuit et local pour la transition de carrière des athlètes de haut niveau**

[Français](#français) | [English](#english)

</div>

---

## Français

### 🎯 Description

Transition Assistant est un chatbot IA moderne qui aide les athlètes de haut niveau dans leur reconversion professionnelle. L'application utilise vos documents Google Drive pour créer une base de connaissances personnalisée et répond aux questions en français et en anglais.

### ✨ Caractéristiques

- **100% Gratuit** : Utilise Ollama (IA locale) et HuggingFace (embeddings gratuits)
- **Bilingue** : Détecte automatiquement la langue et répond en conséquence
- **Design Premium** : Interface glass morphism avec effets 3D
- **Responsive** : Fonctionne sur ordinateur, tablette et mobile
- **Sécurisé** : Vos données restent privées (traitement local)
- **RAG (Retrieval Augmented Generation)** : Réponses basées sur vos documents

### 📋 Prérequis

- Python 3.8 ou supérieur
- 8 GB de RAM minimum (16 GB recommandé)
- Compte Google avec accès Drive
- Connexion internet (uniquement pour Google Drive)

### 🚀 Installation

#### Étape 1 : Cloner le projet

```bash
git clone https://github.com/votre-repo/transition-assistant.git
cd transition-assistant
```

#### Étape 2 : Installer Python et les dépendances

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate

# Installer les packages
pip install -r requirements.txt
```

#### Étape 3 : Installer Ollama

1. Téléchargez Ollama : https://ollama.ai
2. Installez-le sur votre système
3. Ouvrez un terminal et lancez :

```bash
ollama run mistral
```

Attendez que le modèle soit téléchargé (environ 4 GB).

#### Étape 4 : Configurer Google Drive

1. **Créer un projet Google Cloud** :
   - Allez sur [Google Cloud Console](https://console.cloud.google.com)
   - Créez un nouveau projet
   - Activez l'API Google Drive

2. **Créer un compte de service** :
   - Dans le menu, allez dans "APIs & Services" > "Credentials"
   - Cliquez sur "Create Credentials" > "Service Account"
   - Donnez un nom au compte
   - Téléchargez la clé JSON
   - Renommez le fichier en `credentials.json`
   - Placez-le à la racine du projet

3. **Partager votre dossier Drive** :
   - Ouvrez Google Drive
   - Faites un clic droit sur le dossier "Transition"
   - Partagez-le avec l'email du compte de service (dans le JSON)
   - Copiez l'ID du dossier depuis l'URL

4. **Configurer l'application** :
   - Ouvrez `config.py`
   - Remplacez `VOTRE_FOLDER_ID_ICI` par l'ID de votre dossier

### ▶️ Utilisation

1. **Lancer Ollama** (dans un terminal) :
```bash
ollama serve
```

2. **Lancer l'application** (dans un autre terminal) :
```bash
streamlit run app.py
```

3. **Ouvrir dans le navigateur** :
   - L'application s'ouvre automatiquement
   - Sinon, allez sur http://localhost:8501

### 🎨 Personnalisation

#### Changer les couleurs

Modifiez les couleurs dans `config.py` :

```python
THEME_COLORS = {
    "gold": "#FFD700",    # Or
    "silver": "#C0C0C0",  # Argent
    "bronze": "#CD7F32",  # Bronze
}
```

#### Changer le modèle IA

Ollama propose plusieurs modèles gratuits :

```bash
# Modèles disponibles
ollama run llama2      # Plus précis mais plus lent
ollama run codellama   # Pour du code
ollama run vicuna      # Alternative légère
```

Puis modifiez dans `config.py` :

```python
OLLAMA_MODEL = "llama2"
```

### 🔧 Résolution des problèmes

| Problème | Solution |
|----------|----------|
| "Ollama non détecté" | Vérifiez qu'Ollama est lancé : `ollama serve` |
| "Aucun document trouvé" | Vérifiez l'ID du dossier et les permissions |
| "Erreur Google Drive" | Vérifiez le fichier credentials.json |
| Application lente | Réduisez `RETRIEVER_K` dans config.py |
| Mémoire insuffisante | Utilisez un modèle plus petit (vicuna) |

---

## English

### 🎯 Description

Transition Assistant is a modern AI chatbot that helps elite athletes with their career transition. The application uses your Google Drive documents to create a personalized knowledge base and answers questions in both French and English.

### ✨ Features

- **100% Free**: Uses Ollama (local AI) and HuggingFace (free embeddings)
- **Bilingual**: Automatically detects language and responds accordingly
- **Premium Design**: Glass morphism interface with 3D effects
- **Responsive**: Works on desktop, tablet, and mobile
- **Secure**: Your data remains private (local processing)
- **RAG (Retrieval Augmented Generation)**: Answers based on your documents

### 📋 Prerequisites

- Python 3.8 or higher
- Minimum 8 GB RAM (16 GB recommended)
- Google account with Drive access
- Internet connection (only for Google Drive)

### 🚀 Installation

#### Step 1: Clone the project

```bash
git clone https://github.com/your-repo/transition-assistant.git
cd transition-assistant
```

#### Step 2: Install Python and dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Mac/Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

#### Step 3: Install Ollama

1. Download Ollama: https://ollama.ai
2. Install it on your system
3. Open a terminal and run:

```bash
ollama run mistral
```

Wait for the model to download (about 4 GB).

#### Step 4: Configure Google Drive

1. **Create a Google Cloud project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable Google Drive API

2. **Create a service account**:
   - In menu, go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Name the account
   - Download the JSON key
   - Rename file to `credentials.json`
   - Place it at project root

3. **Share your Drive folder**:
   - Open Google Drive
   - Right-click on "Transition" folder
   - Share with service account email (in JSON)
   - Copy folder ID from URL

4. **Configure the application**:
   - Open `config.py`
   - Replace `VOTRE_FOLDER_ID_ICI` with your folder ID

### ▶️ Usage

1. **Start Ollama** (in a terminal):
```bash
ollama serve
```

2. **Start the application** (in another terminal):
```bash
streamlit run app.py
```

3. **Open in browser**:
   - Application opens automatically
   - Otherwise, go to http://localhost:8501

### 🎨 Customization

#### Change colors

Modify colors in `config.py`:

```python
THEME_COLORS = {
    "gold": "#FFD700",    # Gold
    "silver": "#C0C0C0",  # Silver
    "bronze": "#CD7F32",  # Bronze
}
```

#### Change AI model

Ollama offers several free models:

```bash
# Available models
ollama run llama2      # More accurate but slower
ollama run codellama   # For code
ollama run vicuna      # Lightweight alternative
```

Then modify in `config.py`:

```python
OLLAMA_MODEL = "llama2"
```

### 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Ollama not detected" | Check Ollama is running: `ollama serve` |
| "No documents found" | Check folder ID and permissions |
| "Google Drive error" | Check credentials.json file |
| Slow application | Reduce `RETRIEVER_K` in config.py |
| Insufficient memory | Use smaller model (vicuna) |

---

## 📁 Structure du projet / Project Structure

```
transition-assistant/
├── app.py              # Application principale / Main application
├── config.py           # Configuration
├── requirements.txt    # Dépendances Python / Python dependencies
├── credentials.json    # Google Service Account (à ajouter / to add)
├── README.md          # Documentation
└── logs/              # Logs (créé automatiquement / created automatically)
```

## 🤝 Contribution

Les contributions sont bienvenues ! / Contributions are welcome!

1. Fork le projet / Fork the project
2. Créez votre branche / Create your branch (`git checkout -b feature/AmazingFeature`)
3. Committez / Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request / Open a Pull Request

## 📝 License

MIT License - Voir `LICENSE` pour plus de détails / See `LICENSE` for details

## 🙏 Remerciements / Acknowledgments

- [Ollama](https://ollama.ai) - IA locale gratuite / Free local AI
- [Streamlit](https://streamlit.io) - Framework web / Web framework
- [LangChain](https://langchain.com) - RAG framework
- [HuggingFace](https://huggingface.co) - Embeddings gratuits / Free embeddings

## 📧 Contact

Pour toute question / For any questions:
- Email: support@transition-assistant.com
- GitHub Issues: [Créer une issue / Create an issue](https://github.com/your-repo/issues)

---

<div align="center">
  
**Made with ❤️ for Elite Athletes**

🥇 🥈 🥉

</div>
