Image Search Pipeline
A Streamlit-based application that leverages CLIP (ViT-L-14), Google Custom Search API, and OpenAI's GPT-4o to perform advanced image searches based on detailed textual descriptions. The pipeline decomposes descriptions into components, generates search queries, fetches images, and ranks them by relevance using CLIP embeddings.


Description Decomposition: Breaks down complex image descriptions into key components using GPT-4o.
Dynamic Search Queries: Generates multiple search queries with synonyms and contextual keywords.
Image Ranking: Uses CLIP to rank images based on relevance to the description and its components.
Streamlit Interface: User-friendly web interface for inputting descriptions and viewing results.
Caching: Implements caching for text embeddings and search results to improve performance.
Customizable: Allows prioritization of description components and adjustment of search quality.

Prerequisites
Before installing and running the project, ensure you have the following:

Python: Version 3.8 or higher.
Git: Installed for cloning the repository.
API Keys:
Google Custom Search API Key and Search Engine ID: Obtain from Google Cloud Console.
OpenAI API Key: Obtain from OpenAI Platform.


Hardware:
A GPU is recommended for faster CLIP model processing (CUDA-compatible if using GPU).
At least 8GB RAM for model loading and processing.



Installation
Follow these steps to set up the project locally:

Clone the Repository:
git clone https://github.com/Mu240/image-search-pipeline
cd image-search-pipeline


Create a Virtual Environment:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Python Dependencies:Install the required Python packages using the provided requirements.txt:
pip install -r requirements.txt

Set Up Environment Variables:Create a .env file in the project root or set environment variables manually:
echo "GOOGLE_API_KEY=your_google_api_key" >> .env
echo "GOOGLE_CSE_ID=your_google_cse_id" >> .env
echo "OPENAI_API_KEY=your_openai_api_key" >> .env

Alternatively, export them in your terminal:
export GOOGLE_API_KEY=your_google_api_key
export GOOGLE_CSE_ID=your_google_cse_id
export OPENAI_API_KEY=your_openai_api_key


Download NLTK Data:Run the following command to download required NLTK data:
python -c "import nltk; nltk.download('wordnet')"



Usage
To run the application:

Activate the Virtual Environment (if not already activated):
source venv/bin/activate  # On Windows: venv\Scripts\activate


Run the Streamlit App:
streamlit run app.py

This will start a local web server, and a browser window will open at http://localhost:8501.

Interact with the Interface:

Enter an image description (e.g., "A melancholic violinist playing in a rainy Parisian street at dusk").
Adjust the maximum number of results using the slider (1–10).
Use the "Advanced Options" to:
Toggle GPU usage (if available).
Select search quality (Fast, Balanced, High Quality).
Prioritize a description component (e.g., Main Subject, Emotion).


Click "Search Images" to retrieve and display ranked images.


View Results:

Results display with images, relevance scores, and titles.
Expand "Match Details" for additional information like source URL and image ID.

