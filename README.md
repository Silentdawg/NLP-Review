# NLPReview - Analyse des Feedbacks Employés

AI Text Analysis Tool

An interactive desktop application for text analysis and visualization, built with PySide6 and Transformers.
It allows you to upload text files, extract keywords, generate word clouds, visualize sentiment, and export reports in PDF.

✨ Features

📂 Import text files for analysis

🔑 Keyword extraction using KeyBERT

💬 Sentiment analysis with Transformers (Hugging Face models)

☁️ Generate word clouds

📊 Data visualization (charts, graphs, reports)

🖥️ GUI built with PySide6

📑 Export results to PDF reports

🛠️ Installation

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # on Linux/Mac
venv\Scripts\activate      # on Windows


Install dependencies:

pip install -r requirements.txt

🚀 Usage

Run the application:

python main.py


The GUI will open, and you can start uploading text files for analysis.

📂 Project Structure
├── main.py              # Entry point for the app (PySide6 GUI)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── /data                # (Optional) store input text files
├── /reports             # Generated PDF reports
└── /visuals             # Word clouds & plots

🧠 Technologies Used

PySide6 → GUI

Transformers (Hugging Face) → NLP & sentiment analysis

KeyBERT + Sentence-Transformers → Keyword extraction

Matplotlib & Seaborn → Visualizations

WordCloud → Word cloud generation

FPDF → PDF report export

Pandas & NumPy → Data handling

📜 License

This project is licensed under the MIT License – free to use and modify.

## 👥 Contributeurs
- [Silentdawg](https://github.com/Silentdawg)
- [LulDrako](https://github.com/LulDrako)
