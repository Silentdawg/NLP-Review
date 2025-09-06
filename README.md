NLPReview - Analyse des Feedbacks Employés
## 📌 Description  
**AI Text Analysis Tool** est une application interactive d’analyse et de visualisation de textes, construite avec **PySide6** et les modèles **Transformers**.  
Elle permet d’importer des fichiers textes, d’extraire des mots-clés, d’analyser le sentiment, de générer des nuages de mots et d’exporter des rapports en PDF.  

---

## 🛠 Fonctionnalités  
- 📂 **Import de fichiers texte** pour l’analyse  
- 🔑 **Extraction de mots-clés** avec **KeyBERT**  
- 💬 **Analyse de sentiment** via les modèles **Transformers** (Hugging Face)  
- ☁️ **Nuage de mots** interactif  
- 📊 **Visualisation des résultats** (graphiques, tendances, rapports)  
- 🖥 **Interface utilisateur intuitive** avec **PySide6 (Qt)**  
- 📑 **Export en PDF** des résultats d’analyse  

---

## 🏗 Technologies utilisées  
- **Python** : Pandas, NumPy, Matplotlib, Seaborn, WordCloud, scikit-learn  
- **NLP** : Transformers (Hugging Face), Torch, KeyBERT, Sentence-Transformers  
- **Interface utilisateur** : PySide6 (Qt)  
- **Rapports** : FPDF pour l’export en PDF  

---

## 🚀 Installation  

1. **Cloner le dépôt**  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Créer un environnement virtuel (recommandé)**  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Installer les dépendances**  
```bash
pip install -r requirements.txt
```

4. **Lancer l’application**  
```bash
python main.py
```

---

## 📂 Structure du projet  

```
├── main.py              # Point d’entrée de l’application (GUI PySide6)
├── requirements.txt     # Dépendances Python
├── README.md            # Documentation du projet
├── /data                # (Optionnel) Fichiers textes d’entrée
├── /reports             # Rapports PDF générés
├── /visuals             # Graphiques et nuages de mots
└── /assets              # (Optionnel) ressources diverses
```

---

## 📜 Licence  
Ce projet est sous licence **MIT** – voir le fichier [LICENSE](LICENSE) pour plus de détails.  

## 👥 Contributeurs
- [Silentdawg](https://github.com/Silentdawg)
- [LulDrako](https://github.com/LulDrako)
