# 🎬 CineMatch — Movie Recommendation System

A content-based movie recommendation system that suggests similar films based on genres, cast, director, keywords, overview, and tagline. Built with TF-IDF vectorization and cosine similarity, served through a styled Streamlit web app.

---

## 📌 Overview

Most recommendation systems require user history or ratings to work. CineMatch takes a different approach — it understands the **content** of a movie itself and finds others that are genuinely similar in terms of story, people, and style.

Given any movie title from the dataset, the app returns the top N most similar movies ranked by a similarity score, along with their director, cast, and genres.

---

## ❗ Problem

Finding a movie similar to one you already love is harder than it sounds. Streaming platforms bury their recommendation logic behind black-box algorithms optimized for watch time, not genuine content similarity. There's no transparent, simple tool that lets you say *"I liked this film — find me more like it"* and understand **why** those results were chosen.

---

## ✅ Solution

CineMatch combines multiple movie attributes — genres, cast, director, keywords, overview, and tagline — into a single text representation per movie. It then applies **TF-IDF vectorization** to capture the importance of each term across the dataset and computes **cosine similarity** between movies to find the closest matches. The result is a fast, interpretable, content-driven recommender with no black boxes.

---

## ⚙️ How It Works

1. **Feature Engineering** — Combines `genres`, `keywords`, `cast`, `director`, `overview`, `tagline`, and `original_language` into a single text field per movie.
2. **Text Preprocessing** — Lowercasing, punctuation removal, stopword filtering, and lemmatization via NLTK.
3. **TF-IDF Vectorization** — Transforms the combined text into a sparse matrix with up to 50,000 features using 1–2 word n-grams.
4. **Cosine Similarity** — Measures the angle between two movie vectors to determine content closeness.
5. **Pickle Serialization** — The processed dataframe, TF-IDF matrix, and index mapping are saved as `.pkl` files so the Streamlit app loads instantly without reprocessing on every run.

---

## 🗂️ Project Structure

```
cinematch/
│
├── Movie_Recommendation.ipynb   # Full preprocessing + model training notebook
├── app.py                       # Streamlit web application
│
├── movies_df.pkl                # Processed movie dataframe
├── tfidf_matrix.pkl             # Fitted TF-IDF sparse matrix
├── indices.pkl                  # Title → index mapping
│
├── movies.csv                   # Raw dataset (see Dataset section below)
└── requirements.txt             # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cinematch.git
cd cinematch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `movies.csv` and place it in the root directory. The dataset should contain columns: `title`, `genres`, `keywords`, `cast`, `director`, `overview`, `tagline`, `original_language`.

### 4. Generate the pkl files

Run the notebook `Movie_Recommendation.ipynb` top to bottom. The final cells will produce:

```
movies_df.pkl
tfidf_matrix.pkl
indices.pkl
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Live Demo

👉 **[your-app-link.streamlit.app](https://your-app-link.streamlit.app)**

> Replace the link above with your deployed Streamlit Cloud URL.

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
nltk
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Data | Pandas, NumPy |
| NLP | NLTK |
| ML | Scikit-learn (TF-IDF, Cosine Similarity) |
| App | Streamlit |
| Serialization | Pickle |

---

## 🔮 Future Improvements

- Blend collaborative filtering with content signals for hybrid recommendations
- Integrate TMDB API for live posters and movie details
- Support multi-movie input — recommend based on a list of favourites
- Add filters for genre, release year, and language

---

## 📄 License

This project is licensed under the MIT License.
