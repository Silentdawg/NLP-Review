import pandas as pd
import spacy
from collections import Counter
from textblob import TextBlob
import matplotlib.pyplot as plt
import nltk
from langdetect import detect  # To detect language dynamically

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load English & French spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")

# Load Excel Data and strip column names
file_path = "Data_Ey.xlsx"  # Adjust path as needed
df = pd.read_excel(file_path, dtype=str)
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

# Define stopwords for both languages
stopwords_en = set(stopwords.words("english"))
stopwords_fr = set(stopwords.words("french"))

def get_employee_data(employee_id):
    """Fetches data based on Employee ID."""
    id_col = "Feedback Requester GUI"

    if id_col not in df.columns:
        return None, None, "Error: Column 'Feedback Requester GUI' not found in dataset."

    filtered_df = df[df[id_col].astype(str) == str(employee_id)]
    
    if filtered_df.empty:
        return None, None, f"No data found for Employee ID: {employee_id}"

    # Ensure expected columns exist
    required_columns = ["Feedback Dimensions & Questions", "Feedback Responses", "Comments"]
    for col in required_columns:
        if col not in df.columns:
            return None, None, f"Error: Missing necessary column '{col}' in dataset."

    qa_df = filtered_df[["Feedback Dimensions & Questions", "Feedback Responses"]].fillna("")  # Replace NaN
    text_df = filtered_df[["Comments"]].fillna("")  # Replace NaN

    return text_df, qa_df, None

def analyze_qa_answers(qa_df):
    """Analyzes the QA responses (supports both English & French)."""
    if qa_df is None or qa_df.empty:
        return "No QA data available for analysis."

    checked_qa, text_qa = [], []

    for _, row in qa_df.iterrows():
        question, response = row["Feedback Dimensions & Questions"], row["Feedback Responses"]

        # Ensure response is a string (convert float or NaN to empty string)
        if isinstance(response, float):
            response = str(response)  

        if len(response) <= 50:  # Ensure `len()` works
            checked_qa.append((question, response))
        else:
            text_qa.append((question, response))

    # Process checked answers and text responses for sentiment and key ideas
    checked_counter = Counter([resp for _, resp in checked_qa])
    strengths, weaknesses, key_ideas, sentiment_scores = [], [], [], []

    # Track word frequency for strengths & weaknesses
    strength_words = []
    weakness_words = []

    for _, response in text_qa:
        try:
            lang = detect(response)  # Detect if the response is English or French
        except:
            lang = "en"  # Default to English if detection fails

        nlp = nlp_fr if lang == "fr" else nlp_en  # Select correct spaCy model
        stopwords_set = stopwords_fr if lang == "fr" else stopwords_en  # Select stopwords set

        doc = nlp(response)
        sentiment = TextBlob(response).sentiment.polarity
        sentiment_scores.append(sentiment)
        key_ideas.extend([chunk.text.lower() for chunk in doc.noun_chunks])

        words = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in stopwords_set]
        
        if sentiment > 0:
            strengths.append(response)
            strength_words.extend(words)
        elif sentiment < 0:
            weaknesses.append(response)
            weakness_words.extend(words)

    # Count word frequencies for strengths & weaknesses
    strength_counter = Counter(strength_words).most_common(5)
    weakness_counter = Counter(weakness_words).most_common(5)

    # Filter stopwords and count key ideas frequency
    filtered_ideas = [idea for idea in key_ideas if idea not in stopwords_set]
    ideas_counter = Counter(filtered_ideas)
    top_key_ideas = ideas_counter.most_common(5)

    # Visualization for strength & weakness words
    if strength_counter:
        plt.figure(figsize=(8, 4))
        plt.bar([item[0] for item in strength_counter], [item[1] for item in strength_counter], color='green')
        plt.title("Top Strength Words")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    if weakness_counter:
        plt.figure(figsize=(8, 4))
        plt.bar([item[0] for item in weakness_counter], [item[1] for item in weakness_counter], color='red')
        plt.title("Top Weakness Words")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Visualization for checked answers
    if checked_counter:
        plt.figure(figsize=(8, 4))
        plt.bar(checked_counter.keys(), checked_counter.values(), color='blue')
        plt.title("Checked Answers Frequency")
        plt.xlabel("Answer")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Visualization for top key ideas
    if top_key_ideas:
        plt.figure(figsize=(8, 4))
        plt.bar([item[0] for item in top_key_ideas], [item[1] for item in top_key_ideas], color='purple')
        plt.title("Top Key Ideas Frequency")
        plt.xlabel("Key Idea")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    summary = (
        f"Checked Answers Frequency:\n{checked_counter}\n\n"
        f"Strengths: {strengths}\n\n"
        f"Weaknesses: {weaknesses}\n\n"
        f"Top Key Ideas: {top_key_ideas}\n"
        f"Top Strength Words: {strength_counter}\n"
        f"Top Weakness Words: {weakness_counter}\n"
    )
    return summary
