# 🚀 DÉMARRAGE RAPIDE / QUICK START

## 📦 Fichiers créés / Files created:

✅ **Application principale / Main application**
- `app.py` - Application Streamlit avec design glass morphism 3D
- `config.py` - Configuration centralisée

✅ **Installation / Setup**
- `requirements.txt` - Dépendances Python
- `install.sh` - Script d'installation Linux/Mac
- `install.bat` - Script d'installation Windows
- `start.sh` - Lanceur Linux/Mac
- `start.bat` - Lanceur Windows

✅ **Docker (optionnel)**
- `Dockerfile` - Image Docker
- `docker-compose.yml` - Orchestration complète

✅ **Documentation**
- `README.md` - Guide complet bilingue
- `QUICKSTART.md` - Ce fichier

✅ **Sécurité**
- `.gitignore` - Protection des fichiers sensibles
- `credentials.json.example` - Template Google

---

## ⚡ Installation en 5 minutes

### Windows:
```batch
1. Double-cliquez sur install.bat
2. Configurez credentials.json et config.py
3. Double-cliquez sur start.bat
```

### Linux/Mac:
```bash
1. chmod +x install.sh && ./install.sh
2. Configurez credentials.json et config.py  
3. ./start.sh
```

### Docker:
```bash
1. docker-compose up -d
2. Ouvrez http://localhost:8501
```

---

## 🎨 Points forts du design

### Interface Glass Morphism 3D
- **Effet glass** avec backdrop-filter blur
- **Animations 3D** pour les médailles (rotation)
- **Dégradés métalliques** Or/Argent/Bronze
- **Particules flottantes** animées
- **Responsive** mobile-first

### Couleurs personnalisables
```python
# Dans config.py
THEME_COLORS = {
    "gold": "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32"
}
```

---

## 🤖 IA 100% Gratuite

### Technologies utilisées:
- **Ollama** - IA locale (Mistral)
- **HuggingFace** - Embeddings multilingues gratuits
- **FAISS** - Base vectorielle locale
- **LangChain** - Framework RAG

### Pas d'API payantes:
❌ Pas d'OpenAI  
❌ Pas de Claude API  
❌ Pas de services cloud  
✅ 100% local et gratuit!

---

## 🔗 Configuration Google Drive

### 1. Créer un projet Google Cloud
```
1. https://console.cloud.google.com
2. Nouveau projet
3. Activer API Drive
```

### 2. Créer compte de service
```
1. APIs & Services > Credentials
2. Create Credentials > Service Account
3. Télécharger JSON → credentials.json
```

### 3. Partager dossier Drive
```
1. Clic droit sur dossier "Transition"
2. Partager avec email du compte de service
3. Copier l'ID depuis l'URL
```

### 4. Configurer l'app
```python
# Dans config.py
GOOGLE_DRIVE_FOLDER_ID = "votre_id_ici"
```

---

## 🌍 Fonctionnalités bilingues

### Détection automatique de langue:
- Questions en français → Réponses en français
- Questions in English → Answers in English
- Mélange des langues supporté

### Base de connaissances:
- Documents Word (.docx, .doc)
- Multilingue dans le même dossier
- Mise à jour en temps réel

---

## 📱 Design Responsive

### Mobile (< 480px)
- Interface adaptée tactile
- Taille de police optimisée
- Médailles réduites

### Tablette (< 768px)
- Disposition flexible
- Chat plein écran

### Desktop (> 768px)
- Effets 3D complets
- Animations avancées

---

## 🆘 Dépannage rapide

| Problème | Solution |
|----------|----------|
| Ollama non détecté | `ollama serve` dans terminal |
| Erreur Google Drive | Vérifier credentials.json |
| Application lente | Réduire RETRIEVER_K = 2 |
| Mémoire insuffisante | Utiliser modèle vicuna |

---

## 💡 Tips Pro

### Performance:
```python
# config.py
CHUNK_SIZE = 500  # Plus petit = plus rapide
RETRIEVER_K = 2   # Moins de contexte = plus rapide
```

### Qualité:
```python
# config.py  
CHUNK_SIZE = 1500  # Plus grand = meilleur contexte
RETRIEVER_K = 5    # Plus de résultats = réponses complètes
```

### Modèles alternatifs:
```bash
ollama run llama2    # Plus précis
ollama run vicuna    # Plus léger
ollama run mixtral   # Multilingue++
```

---

## 📞 Support

**Email**: support@transition-assistant.com  
**GitHub**: github.com/your-repo/issues  
**Documentation**: Voir README.md complet

---

## 🎯 Prêt à démarrer?

1. **Installation**: 5 minutes ⏱️
2. **Configuration**: 5 minutes ⚙️
3. **Utilisation**: Immédiate! 🚀

**Bonne transition! / Good transition!** 🏅
