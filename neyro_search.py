from sentence_transformers import SentenceTransformer, util
import mysql.connector
import torch
import sys
import json
import os
import re
import math
import pickle
import time
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    torch.set_grad_enabled(False)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model loaded in {time.time() - start_time:.2f} seconds on {device}")
    
    get_products_with_embeddings()
    print("Embeddings loaded and ready")
    
    yield  

app = FastAPI(
    title="Product Search API", 
    description="Search products",
    lifespan=lifespan
)

model = None

@lru_cache(maxsize=1)
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        port=3333,
        user="root",
        password="admin",
        database="auth_db",
        auth_plugin='mysql_native_password',
        buffered=True
    )

# Improved embeddings caching
EMBEDDINGS_CACHE_FILE = 'product_embeddings_cache.pkl'

def get_products_with_embeddings():
    cache_time = time.time()
    
    cache_exists = os.path.exists(EMBEDDINGS_CACHE_FILE)
    cache_timestamp = os.path.getmtime(EMBEDDINGS_CACHE_FILE) if cache_exists else 0
   
    
    # Check cache freshness (once a day - 86400 seconds)
    cache_age = time.time() - cache_timestamp
    print(f"Cache exists: {cache_exists}, Cache age: {cache_age}")
    force_refresh = cache_age > 86400
    
    if cache_exists and not force_refresh:
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data
        except Exception as e:
            print(f"Error loading cache: {e}")
    db_time = time.time()
    
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT id, title, description FROM posts")
    products = cursor.fetchall()
    cursor.close()
    
    if not products:
        return [], [], []
    
    product_ids = [product[0] for product in products]
    product_texts = []
    product_lengths = []
    
    for product in products:
        product_text = f"{product[1]} {product[2]}"
        product_texts.append(product_text)
        product_lengths.append(len(product_text.split()))
    
    embedding_time = time.time()
    batch_size = 64  
    product_embeddings = []
    
    for i in range(0, len(product_texts), batch_size):
        batch = product_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
        product_embeddings.append(batch_embeddings)
    
    product_embeddings = torch.cat(product_embeddings, dim=0)
    
    cache_data = (product_ids, product_embeddings, product_lengths)
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    return cache_data

def search_products(user_query, top_n=20):
    search_start = time.time()
    
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    
    product_ids, product_embeddings, product_lengths = get_products_with_embeddings()
    
    if not product_ids:
        return []
    similarities = util.cos_sim(query_embedding, product_embeddings)[0]
    SIMILARITY_THRESHOLD = 0.2
    results = []
    for idx, similarity in enumerate(similarities):
        similarity_score = similarity.item()
        if similarity_score > SIMILARITY_THRESHOLD:
            results.append({
                'id': product_ids[idx],
                'similarity': similarity_score
            })
    
    # Sort and take top-N
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = results[:top_n]
    
    print(f"Search completed in {time.time() - search_start:.2f} seconds")
    return top_results

class SearchRequest(BaseModel):
    query: str
    top_n: int = 20

@app.post("/search")
async def api_search_products(request: SearchRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    start_time = time.time()
    results = search_products(request.query, request.top_n)
    
    return {
        "results": results
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model": "intfloat/multilingual-e5-large"}

if __name__ == "__main__":
    uvicorn.run("neyro_search:app", host="localhost", port=8000, reload=False)
