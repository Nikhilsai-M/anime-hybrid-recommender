import pickle
import json
import os
import re
import numpy as np
import faiss
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Initialize the API
app = FastAPI()

# Allow the frontend to talk to us (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. IMAGE CACHE SYSTEM
# ==========================================
# I'm using a local JSON file to map Anime IDs to Poster URLs.
# This prevents us from hitting external APIs constantly.
IMAGE_CACHE_FILE = "anime_images_cache.json"
poster_map = {}

@app.on_event("startup")
async def startup_event():
    global poster_map
    print("📂 Loading Image Cache...")
    
    if os.path.exists(IMAGE_CACHE_FILE):
        try:
            with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse the JSON to build a fast lookup dictionary
            for item in data.get('data', []):
                picture = item.get('picture')
                if not picture: continue
                
                # Extract ID from the MyAnimeList URL
                for source in item.get('sources', []):
                    if "myanimelist.net/anime/" in source:
                        try:
                            match = re.search(r'/anime/(\d+)', source)
                            if match:
                                poster_map[match.group(1)] = picture
                                break
                        except: pass
        except Exception as e:
            print(f"⚠️ Warning: Could not load image cache: {e}")
            
    print(f"✅ Loaded {len(poster_map)} posters.")

def get_poster(anime_id):
    # Fallback to a placeholder if we don't have the image
    return poster_map.get(str(anime_id), "https://via.placeholder.com/225x320?text=No+Image")

# ==========================================
# 2. LOAD AI MODELS
# ==========================================
print("🚀 Loading AI Models...")

# A. Load the Metadata (Encoders & Dataframes)
with open('production_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

anime_df = meta['anime_df']
user_enc = meta['user_enc']
item_enc = meta['item_enc']

# B. Load the Semantic Search Engine (BERT + FAISS)
# This handles the "What is this anime about?" part.
vector_index = faiss.read_index("anime_vector_db.index")
print("🧠 Loading BERT...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# C. REBUILD THE NEURAL NETWORK
# I am manually reconstructing the model architecture here to avoid 
# version conflicts between Colab (Keras 3) and Local (Keras 2).
print("🔨 Re-building Neural Network Structure...")

num_users = len(user_enc.classes_)
num_items = len(item_enc.classes_)
embedding_size = 32

# The Architecture:
# We take a User ID and an Item ID, embed them into vectors,
# and pass them through dense layers to predict a rating.
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
item_embedding = Embedding(num_items, embedding_size, name='item_embedding')(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])
fc1 = Dense(128, activation='relu')(concat)
fc2 = Dense(64, activation='relu')(fc1)
output = Dense(1, activation='sigmoid')(fc2) # Output is a probability (0-1)

model = Model([user_input, item_input], output)

# D. Load the Weights
print("🧠 Loading Model Weights (.h5)...")
model.load_weights('anime_neumf_model.h5')

print("✅ System Ready")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
class RequestModel(BaseModel):
    user_id: str
    query: str

def predict_score(user_id, candidate_indices):
    """
    Given a user and a list of anime indices, predicts how much 
    the user will like each anime.
    """
    # Check if the user exists in our training data
    if user_id.isdigit() and int(user_id) in user_enc.classes_:
        u_val = int(user_id)
        u_enc_id = user_enc.transform([u_val])[0]
        is_guest = False
    else:
        # Default to User 0 (or generic behavior) for guests
        u_enc_id = 0
        is_guest = True

    # Get the internal Anime IDs for the candidates
    cand_ids = anime_df.iloc[candidate_indices]['anime_id'].values
    
    # Filter out any anime that wasn't in our training set
    valid_mask = np.isin(cand_ids, item_enc.classes_)
    valid_cand_ids = cand_ids[valid_mask]

    scores = np.zeros(len(candidate_indices))
    
    if len(valid_cand_ids) > 0:
        c_enc_ids = item_enc.transform(valid_cand_ids)
        
        # Create input array for the model: [User, User, ...]
        user_in = np.array([u_enc_id] * len(c_enc_ids))
        
        # Ask the Brain!
        preds = model.predict([user_in, c_enc_ids], verbose=0).flatten()
        scores[valid_mask] = preds
        
    return scores, is_guest

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/home/{user_id}")
def get_home(user_id: str):
    """
    Returns a personalized feed.
    We take the top 100 popular anime and re-rank them based on user preference.
    """
    indices_pool = list(range(0, 100)) 
    scores, is_guest = predict_score(user_id, indices_pool)
    
    items = []
    for i, idx in enumerate(indices_pool):
        aid = int(anime_df.iloc[idx]['anime_id'])
        items.append({
            "id": aid,
            "title": anime_df.iloc[idx]['Name'],
            "score": float(scores[i]),
            "image": get_poster(aid)
        })
    
    # If it's a known user, sort by their predicted score.
    # If guest, leave it as default popularity order.
    if not is_guest:
        items.sort(key=lambda x: x['score'], reverse=True)
        
    return items[:18] # Sending 18 items to fill wider screens

@app.post("/search")
def search_anime(req: RequestModel):
    """
    Semantic Search using BERT.
    Finds anime by meaning, not just keywords.
    """
    query = req.query
    
    # Convert text query -> Vector
    query_vec = bert_model.encode([query])
    faiss.normalize_L2(query_vec)
    
    # Search the Vector DB
    dist, search_indices = vector_index.search(query_vec, 12)
    
    items = []
    for i in search_indices[0]:
        if i < 0 or i >= len(anime_df): continue
        aid = int(anime_df.iloc[i]['anime_id'])
        items.append({
            "id": aid,
            "title": anime_df.iloc[i]['Name'],
            "match_score": float(dist[0][list(search_indices[0]).index(i)]),
            "image": get_poster(aid)
        })
    return items

@app.post("/recommend")
def get_recommendations(req: RequestModel):
    """
    The Hybrid Recommender.
    Returns two lists:
    1. Content Match (Left): Purely similar plots (BERT).
    2. AI Picks (Right): Similar plots re-ranked by User Preference (BERT + NCF).
    """
    query = req.query
    query_vec = bert_model.encode([query])
    faiss.normalize_L2(query_vec)
    
    # 1. Deep Search: Get 500 candidates to ensure variety
    dist, search_indices = vector_index.search(query_vec, 500)
    
    candidates = search_indices[0]
    sim_scores = dist[0]
    
    # 2. Get User Preferences for these candidates
    ai_scores, is_guest = predict_score(req.user_id, candidates)
    
    left = []
    right = []
    seen = set()
    
    query_clean = query.lower().strip()
    franchise_count = 0  
    FRANCHISE_LIMIT = 4  # Cap to prevent "Movie 1, Movie 2, Movie 3" spam

    for i, db_idx in enumerate(candidates):
        if db_idx < 0 or db_idx >= len(anime_df): continue
        name = anime_df.iloc[db_idx]['Name']
        aid = int(anime_df.iloc[db_idx]['anime_id'])
        name_clean = name.lower().strip()
        
        # Don't recommend the show itself
        if name == query: continue
        
        # Don't show duplicates
        if name in seen: continue
        
        # Franchise Logic: Allow a few sequels, then force variety
        is_franchise = query_clean in name_clean
        if is_franchise:
            if franchise_count >= FRANCHISE_LIMIT:
                continue
            franchise_count += 1
        
        seen.add(name)
        
        item = {
            "id": aid, 
            "title": name, 
            "match": float(sim_scores[i]),
            "ai": float(ai_scores[i]),
            "image": get_poster(aid)
        }
        
        # Add to "Content Match" list
        left.append(item)
        
        # Add to "AI Picks" list (Strictly NO franchise spam here)
        if not is_franchise:
            item_right = item.copy()
            # Hybrid Score: 10% similarity + 90% user preference
            item_right["hybrid"] = (item["match"] * 0.1) + (item["ai"] * 0.9)
            right.append(item_right)
            
    # Sort the lists
    left.sort(key=lambda x: x['match'], reverse=True)
    right.sort(key=lambda x: x.get('hybrid', 0), reverse=True)
    
    return {"content": left[:24], "ai": right[:24]}