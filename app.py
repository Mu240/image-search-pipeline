import streamlit as st
import os
import json
import requests
from typing import List, Dict, Tuple
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import torch
from PIL import Image
import open_clip
from io import BytesIO
import uuid
import concurrent.futures
import time
import functools
import hashlib
from pathlib import Path
import pickle
import nltk
from nltk.corpus import wordnet
import pandas as pd
import numpy as np

# Download WordNet data (run once)
nltk.download('wordnet')

# Load API keys from environment variables with fallbacks
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCfmkFtQKPQ21ltQWKEgFBVdh1MBm0vz_w")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "e24ceae9dec244e58")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key")

# Constants
CACHE_DIR = Path(".cache")
CACHE_EXPIRY = 86400  # 24 hours in seconds
FLICKR8K_DATA_PATH = Path("flickr8k")  # Adjust to your local path where Flickr8k is stored
FLICKR8K_CAPTIONS = FLICKR8K_DATA_PATH / "captions.txt"
FLICKR8K_IMAGES = FLICKR8K_DATA_PATH / "Images"

class ImageSearchPipeline:
    def __init__(self, google_api_key: str, google_cse_id: str, openai_api_key: str):
        """Initialize pipeline with API keys and setup OpenCLIP ViT-L-14 model."""
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Create cache directory
        CACHE_DIR.mkdir(exist_ok=True)

        # Initialize OpenCLIP ViT-L-14 model with OpenAI pretrained weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='openai', device=self.device
            )
            self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')
        except Exception as e:
            st.error(f"Failed to initialize CLIP model: {str(e)}")
            raise RuntimeError("Model initialization failed")

        # Setup LangChain components
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.decompose_prompt = PromptTemplate(
            input_variables=["description"],
            template="""
            Break down this image description into key components:
            '{description}'

            Return a JSON object with:
            - main_subject: Primary person/animal/object
            - action: What the subject is doing
            - objects: Notable objects involved
            - setting: Location or environment
            - background: Background elements
            - time_period: Era or time context (if specified)
            - emotion: Emotional tone or mood (e.g., joyful, somber)
            - theme: Overarching theme or concept (e.g., adventure, solitude)
            - attributes: Specific visual characteristics (e.g., vibrant colors, dark lighting)
            """
        )
        self.decompose_chain = (
            {"description": RunnablePassthrough()}
            | self.decompose_prompt
            | self.llm
            | StrOutputParser()
        )

        # Cache for text embeddings
        self.text_embedding_cache = {}

        # Default component weights
        self.component_weights = {
            "main_subject": 0.5,
            "action": 0.2,
            "setting": 0.15,
            "objects": 0.1,
            "background": 0.05,
            "emotion": 0.1,
            "theme": 0.1,
            "attributes": 0.05
        }

    @staticmethod
    def get_cache_key(data: str) -> str:
        """Generate a cache key from input data."""
        return hashlib.md5(data.encode()).hexdigest()

    def cache_result(self, cache_key: str, data, cache_type: str):
        """Cache data to disk."""
        try:
            cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump((time.time(), data), f)
        except Exception as e:
            print(f"Error caching {cache_type} result: {e}")

    def get_cached_result(self, cache_key: str, cache_type: str):
        """Retrieve cached data if it exists and is not expired."""
        try:
            cache_file = CACHE_DIR / f"{cache_type}_{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    timestamp, data = pickle.load(f)
                    if time.time() - timestamp < CACHE_EXPIRY:
                        if cache_type == "img_embed" and not isinstance(data, torch.Tensor):
                            data = torch.tensor(data).to(self.device)
                        return data
        except Exception as e:
            print(f"Error retrieving cached {cache_type} result: {e}")
        return None

    @functools.lru_cache(maxsize=128)
    def decompose_description(self, description: str) -> Dict:
        """Break down description into components, with caching."""
        cache_key = self.get_cache_key(description)
        cached_result = self.get_cached_result(cache_key, "decompose")

        if cached_result:
            return cached_result

        try:
            result = self.decompose_chain.invoke(description)
            print(f"Raw LLM response: {result}")
            if not result.strip():
                raise ValueError("Empty response from LLM")

            # Clean markdown code block
            cleaned_result = result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result.replace("```json", "", 1).strip()
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result.rsplit("```", 1)[0].strip()

            # Parse cleaned JSON with robust error handling
            try:
                parsed = json.loads(cleaned_result)
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response is not a valid JSON object")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return {}

            # Cache result
            self.cache_result(cache_key, parsed, "decompose")
            return parsed
        except Exception as e:
            print(f"Error decomposing description: {e}")
            return {}

    def fetch_contextual_keywords(self, description: str) -> List[str]:
        """Fetch relevant keywords from a web search to enrich description."""
        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": description,
                "num": 3,
                "safe": "active"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("items", [])

            # Extract keywords from titles and snippets
            keywords = []
            for item in results:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                words = (title + " " + snippet).split()
                keywords.extend([w.lower() for w in words if len(w) > 4 and w.isalpha()])

            # Return top 5 unique keywords
            return list(set(keywords))[:5]
        except Exception as e:
            print(f"Error fetching contextual keywords: {e}")
            return []

    def generate_search_queries(self, components: Dict) -> List[str]:
        """Generate multiple search queries from components with synonyms and context."""
        queries = []
        main_subject = components.get('main_subject', '')
        action = components.get('action', '')
        setting = components.get('setting', '')
        emotion = components.get('emotion', '')
        theme = components.get('theme', '')

        # Fetch contextual keywords
        description = f"{main_subject} {action} {setting}"
        contextual_keywords = self.fetch_contextual_keywords(description)

        # Primary query
        primary = f"{main_subject} {action} {setting}"
        queries.append(primary.strip())

        # Secondary query with objects and background
        objects = components.get('objects', '')
        if isinstance(objects, list):
            objects = ' '.join(objects)
        background = components.get('background', '')
        if objects or background:
            secondary = f"{main_subject} {objects} {background}"
            queries.append(secondary.strip())

        # Emotion and theme-based query
        if emotion or theme:
            thematic_query = f"{main_subject} {emotion} {theme} {setting}"
            queries.append(thematic_query.strip())

        # Context-enriched query
        if contextual_keywords:
            context_query = f"{main_subject} {action} {setting} {' '.join(contextual_keywords[:2])}"
            queries.append(context_query.strip())

        # Synonym-based queries
        def get_synonyms(word: str) -> List[str]:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
            return list(synonyms)[:2]

        if main_subject:
            for synonym in get_synonyms(main_subject.split()[-1]):
                synonym_query = f"{synonym} {action} {setting}"
                queries.append(synonym_query.strip())
        if setting:
            for synonym in get_synonyms(setting.split()[-1]):
                synonym_query = f"{main_subject} {action} {synonym}"
                queries.append(synonym_query.strip())

        # Remove empty or duplicate queries
        return list(set([q for q in queries if q]))

    def search_images(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for images using Google Custom Search API with caching."""
        cache_key = self.get_cache_key(f"{query}_{max_results}")
        cached_result = self.get_cached_result(cache_key, "search")

        if cached_result:
            return cached_result

        try:
            url = f"https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "searchType": "image",
                "num": min(max_results, 10),
                "imgSize": "large",
                "safe": "active"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("items", [])
            image_results = [{"url": item["link"], "title": item["title"]} for item in results]

            # Cache result
            self.cache_result(cache_key, image_results, "search")
            return image_results
        except Exception as e:
            print(f"Error searching images: {e}")
            return []

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """Get CLIP text embedding with caching."""
        if text in self.text_embedding_cache:
            return self.text_embedding_cache[text]

        text_input = self.clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self.text_embedding_cache[text] = text_features
        return text_features

    def get_image_embedding(self, image_url: str) -> torch.Tensor:
        """Get CLIP image embedding with error handling."""
        cache_key = self.get_cache_key(image_url)
        cached_result = self.get_cached_result(cache_key, "img_embed")

        if cached_result is not None:
            return cached_result

        try:
            response = requests.get(image_url, timeout=5, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            self.cache_result(cache_key, image_features.cpu().numpy(), "img_embed")
            return image_features
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return torch.zeros((1, 768), device=self.device)  # ViT-L-14 embedding size

    def get_image_embedding_local(self, image_path: str) -> torch.Tensor:
        """Get CLIP image embedding for a local image file."""
        try:
            img = Image.open(image_path).convert("RGB")
            image_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features
        except Exception as e:
            print(f"Error processing local image {image_path}: {e}")
            return torch.zeros((1, 768), device=self.device)

    def score_image_pair(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """Score image relevance using pre-computed embeddings."""
        try:
            with torch.no_grad():
                similarity = (image_embedding @ text_embedding.T).item()
            return similarity
        except Exception as e:
            print(f"Error scoring image: {e}")
            return 0.0

    def process_image_batch(self, images: List[Dict], description: str, components: Dict) -> List[Dict]:
        """Process a batch of images in parallel and return ranked results."""
        description_embedding = self.get_text_embedding(description)

        component_embeddings = {}
        for component, content in components.items():
            if content:
                component_text = content
                if isinstance(content, list):
                    component_text = " ".join(content)
                component_embeddings[component] = self.get_text_embedding(component_text)

        # Dynamic component weights based on context
        weights = self.component_weights.copy()
        desc_length = len(description.split())
        if desc_length > 20:  # Long descriptions emphasize setting/emotion
            weights["setting"] += 0.1
            weights["emotion"] += 0.05
            weights["main_subject"] -= 0.15
        if components.get('emotion') or components.get('theme'):
            weights["emotion"] += 0.05
            weights["theme"] += 0.05
            weights["objects"] -= 0.1

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        processed_images = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_image = {
                executor.submit(self.get_image_embedding, img["url"]): img
                for img in images
            }

            for future in concurrent.futures.as_completed(future_to_image):
                img = future_to_image[future]
                try:
                    image_embedding = future.result()
                    if torch.all(image_embedding == 0):
                        continue

                    clip_score = self.score_image_pair(image_embedding, description_embedding)
                    component_score = 0

                    for component, weight in weights.items():
                        if component in component_embeddings:
                            component_similarity = self.score_image_pair(
                                image_embedding,
                                component_embeddings[component]
                            )
                            component_score += component_similarity * weight

                    final_score = (0.5 * clip_score) + (0.5 * component_score)

                    processed_images.append({
                        "url": img["url"],
                        "title": img["title"],
                        "score": final_score,
                        "image_id": str(uuid.uuid4())
                    })
                except Exception as e:
                    print(f"Error processing {img['url']}: {e}")

        return sorted(processed_images, key=lambda x: x["score"], reverse=True)

    def process_description(self, description: str, max_results: int = 5) -> List[Dict]:
        """Main pipeline method to process description and return ranked images."""
        try:
            start_time = time.time()

            # Step 1: Decompose description
            components = self.decompose_description(description)
            if not components:
                return []

            print(f"Decomposition took {time.time() - start_time:.2f} seconds")
            query_start = time.time()

            # Step 2: Generate search queries
            queries = self.generate_search_queries(components)

            # Step 3: Search images for each query
            all_images = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
                future_to_query = {
                    executor.submit(self.search_images, query, max_results): query
                    for query in queries
                }
                for future in concurrent.futures.as_completed(future_to_query):
                    images = future.result()
                    all_images.extend(images)

            print(f"Image search took {time.time() - query_start:.2f} seconds")
            ranking_start = time.time()

            # Remove duplicates
            unique_images = []
            seen_urls = set()
            for img in all_images:
                if img["url"] not in seen_urls:
                    unique_images.append(img)
                    seen_urls.add(img["url"])

            # Step 4: Rank images
            ranked_images = self.process_image_batch(unique_images, description, components)

            print(f"Image ranking took {time.time() - ranking_start:.2f} seconds")
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")

            return ranked_images[:max_results]
        except Exception as e:
            print(f"Error processing description: {e}")
            return []

# Streamlit UI
st.title("Enhanced Image Search Pipeline - Powered by CLIP")
st.markdown("Search for images using detailed descriptions.")

# Input for image description
description = st.text_area(
    "Enter Image Description",
    placeholder="e.g., A melancholic violinist playing in a rainy Parisian street at dusk",
    height=100
)

# Slider for max results
max_results = st.slider("Maximum Number of Results", min_value=1, max_value=10, value=3, key="search_max_results")

# Advanced options
with st.expander("Advanced Options"):
    use_gpu = st.checkbox("Use GPU if available", value=True, key="search_use_gpu")
    search_quality = st.select_slider(
        "Search Quality",
        options=["Fast", "Balanced", "High Quality"],
        value="Balanced",
        help="Fast: Fewer queries, quicker results. High Quality: More queries, better accuracy.",
        key="search_quality"
    )
    prioritize_component = st.selectbox(
        "Prioritize Component",
        options=["None", "Main Subject", "Action", "Setting", "Emotion", "Theme", "Attributes"],
        help="Emphasize a specific component in scoring for tailored results.",
        key="search_prioritize"
    )

# Button to trigger search
if st.button("Search Images", key="search_button"):
    if not description:
        st.error("Please enter an image description.")
    else:
        try:
            pipeline = ImageSearchPipeline(
                google_api_key=GOOGLE_API_KEY,
                google_cse_id=GOOGLE_CSE_ID,
                openai_api_key=OPENAI_API_KEY
            )

            if not use_gpu:
                pipeline.device = "cpu"
                pipeline.clip_model = pipeline.clip_model.to("cpu")

            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            status_text.text("Analyzing description...")
            progress_bar.progress(10)

            # Decompose description
            components = pipeline.decompose_description(description)
            if not components:
                st.error("Failed to decompose description. Please try again.")
                st.stop()

            progress_bar.progress(25)

            # Display decomposed components
            with st.expander("Decomposed Description"):
                st.json(components)

            status_text.text("Searching for images...")
            progress_bar.progress(40)

            # Adjust max_results based on search quality
            quality_map = {"Fast": 5, "Balanced": 10, "High Quality": 15}
            search_max = quality_map[search_quality]

            # Generate queries and search
            queries = pipeline.generate_search_queries(components)
            all_images = []
            for i, query in enumerate(queries):
                images = pipeline.search_images(query, search_max)
                all_images.extend(images)
                progress_bar.progress(40 + (i + 1) * 20 // len(queries))

            unique_images = []
            seen_urls = set()
            for img in all_images:
                if img["url"] not in seen_urls:
                    unique_images.append(img)
                    seen_urls.add(img["url"])

            status_text.text("Ranking images by relevance...")
            progress_bar.progress(70)

            # Adjust weights based on user prioritization
            if prioritize_component != "None":
                component_key = prioritize_component.lower().replace(' ', '_')
                if component_key in pipeline.component_weights:
                    pipeline.component_weights[component_key] += 0.2
                    total_weight = sum(pipeline.component_weights.values())
                    pipeline.component_weights = {k: v / total_weight for k, v in pipeline.component_weights.items()}

            ranked_images = pipeline.process_image_batch(unique_images, description, components)
            progress_bar.progress(90)

            results = ranked_images[:max_results]
            progress_bar.progress(100)

            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

            execution_time = time.time() - start_time
            st.success(f"Found {len(results)} images in {execution_time:.2f} seconds!")

            if results:
                for i, result in enumerate(results, 1):
                    st.subheader(f"Result {i}")
                    st.image(result["url"], caption=f"Score: {result['score']:.2f} | Title: {result['title']}")
                    with st.expander(f"Match Details for Image {i}"):
                        st.write(f"Relevance Score: {result['score']:.4f}")
                        st.write(f"Source URL: {result['url']}")
                        st.write(f"Image ID: {result['image_id']}")
            else:
                st.warning("No images found. Try a different description or adjust settings.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")