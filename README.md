# AI Text Analysis Tool  

An interactive desktop application for text analysis and visualization, built with **PySide6** and **Transformers**.  
It allows you to upload text files, extract keywords, perform sentiment analysis, generate word clouds, and export PDF reports.  

---

## 📌 Description  
**AI Text Analysis Tool** is designed to simplify text analytics through an intuitive interface.  
It integrates modern **NLP models** and visualization libraries to deliver insights from employee feedback, reviews, or any large set of textual data.  

---

## 🛠 Features  
- 📂 **Import text files** for analysis  
- 🔑 **Keyword extraction** with **KeyBERT**  
- 💬 **Sentiment analysis** using **Transformers** (Hugging Face models)  
- ☁️ **Word cloud generation** for quick insights  
- 📊 **Data visualization** (graphs, charts, summaries)  
- 🖥 **User-friendly GUI** built with **PySide6 (Qt)**  
- 📑 **PDF export** of results and reports  

---

## 🏗 Technologies Used  
- **Python**: Pandas, NumPy, Matplotlib, Seaborn, WordCloud, scikit-learn  
- **NLP**: Transformers (Hugging Face), Torch, KeyBERT, Sentence-Transformers  
- **GUI**: PySide6 (Qt)  
- **Reports**: FPDF for PDF export  

---

## 🚀 Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Create a virtual environment (recommended)**  
```bash
python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Run the application**  
```bash
python main.py
```

---

## 📂 Project Structure  

```
├── main.py              # Entry point of the application (PySide6 GUI)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── /data                # (Optional) input text files
├── /reports             # Generated PDF reports
├── /visuals             # Graphs and word clouds
└── /assets              # Assets (icons, resources, etc.)
```

---

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
