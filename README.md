# 🍿 **AnimeStream: Hybrid AI Recommender System**
> *A Next-Gen Recommendation Engine powered by Deep Learning (NCF) and Semantic Search (BERT).*

---

### 🔗 **Live Project Links**
* **Frontend (App):** [https://anime-hybrid-recommender.vercel.app/](https://anime-hybrid-recommender.vercel.app/)
* **Backend (API):** [Hugging Face Space](https://huggingface.co/spaces)

---

### 💡 **Project Overview**
Most recommender systems rely on just one strategy. **AnimeStream uses two brains.**
We implemented a **Hybrid Architecture** that decouples "Content Understanding" from "User Preference":

1.  **Semantic Search (BERT):** Uses `all-MiniLM-L6-v2` to understand the *meaning* of a query (e.g., searching "Time travel thriller" finds *Steins;Gate*).
2.  **Collaborative Filtering (NCF):** A Deep Learning model that predicts user ratings based on interaction history.
3.  **Hybrid Reranking:** Merges content similarity scores with predicted user preference scores to deliver the perfect recommendation.

---

### 🚀 **Key Features**
* **🧠 "Smart" Search:** Finds anime by plot description using Vector Embeddings.
* **🔥 Personalized Feed:** Ranks trending anime specifically for the active user using a Neural Network.
* **⚖️ Dual-Engine Recommendations:**
    * **Left Box:** Pure Content Match (Plot Similarity).
    * **Right Box:** AI Pick (Plot + User Compatibility Score).
* **🚫 Franchise Filtering:** Smart logic to prevent "spammy" recommendations (e.g., filters out excessive movie sequels).
* **⚡ Cinematic UI:** A Netflix-style responsive frontend built with React & Vite.

---

### 🏗️ **System Architecture**

Our Recommender System is built on a **Hybrid Dual-Engine Architecture**. This means we decouple the "Understanding of Content" from the "Understanding of Users" and merge them at the final stage.

#### **1. High-Level Data Flow**

```mermaid
graph LR
    A[User Query] -->|BERT Model| B(Vector Embedding)
    B -->|FAISS Search| C{Semantic Candidates}
    D[User ID] -->|Neural Network| E(User Preference Score)
    C -->|Merge & Rank| F[Final Hybrid Recs]
    E -->|Merge & Rank| F
 ```

## 🧠 The Two "Brains" of the System

### 🔹 Brain A: Semantic Engine
- **Goal:** Find anime with similar plots  
- **Model:** BERT (`all-MiniLM-L6-v2`)
- **Technique:** Vector Embeddings (384-dim)
- **Storage:** FAISS Vector Index

### 🔹 Brain B: Collaborative Engine
- **Goal:** Predict if User X likes Item Y
- **Model:** Neural Collaborative Filtering (NCF)
- **Technique:** Matrix Factorization (Deep Learning)
- **Storage:** TensorFlow `.h5` weights

---

## 🛠️ Tech Stack

| Component        | Technology              | Description |
|------------------|-------------------------|-------------|
| Frontend         | React.js + Vite         | Responsive Glassmorphism UI |
| Backend          | FastAPI                 | High-performance Python API |
| AI Model 1       | Sentence-Transformers   | BERT model for Semantic Search |
| AI Model 2       | TensorFlow / Keras      | Neural Collaborative Filtering (NCF) |
| Vector Database  | FAISS                   | Facebook AI Similarity Search (Sub-millisecond lookup) |

---

## ☁️ Deployment Strategy

Due to the high memory requirements of running **BERT** and **TensorFlow** simultaneously, a **decoupled deployment strategy** is used:

- **Frontend:**  
  Deployed on **Vercel** for fast edge caching.

- **Backend:**  
  Dockerized and deployed on **Hugging Face Spaces (CPU Basic Tier)** to leverage **16GB RAM** for AI models.

---

## 📂 Repository Structure

```plaintext
├── backend/
│   ├── main.py                # FastAPI Server & Logic
│   ├── anime_neumf_model.h5   # Trained NCF Model
│   ├── anime_vector_db.index  # FAISS Vector Index
│   └── requirements.txt       # Python Dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main UI Logic
│   │   └── App.css            # Styling
│   └── package.json           # JS Dependencies
│
└── notebooks/
    └── Model_Training.ipynb   # Research & Training Code
```
