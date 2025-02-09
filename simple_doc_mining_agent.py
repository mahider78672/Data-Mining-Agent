import openai
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")  # Load from environment variable

# Download necessary NLTK resources if not already downloaded
nltk.data.path.append("./nltk_data/")
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir="./nltk_data/")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to read a document
def read_document(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        if not content:
            print(f"Warning: {file_path} is empty.")
            return None
        print(f"Loaded document: {file_path}")
        return content
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

# Preprocessing function (lemmatization instead of stemming)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# Retrieve relevant documents based on cosine similarity
def retrieve_relevant_documents(query, documents, vectorizer):
    query_vector = vectorizer.transform([query])
    tfidf_matrix = vectorizer.transform(documents)
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    relevant_doc_index = np.argmax(cosine_similarities)
    return documents[relevant_doc_index], cosine_similarities[0][relevant_doc_index]

# LLM-based answer generation
def generate_answer_with_llm(query, relevant_doc):
    prompt = f"Question: {query}\n\nRelevant Document: {relevant_doc}\n\nAnswer:"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return "Error generating response."

# Main execution function
def main():
    print("Initializing Document Mining Agent...")

    # Read documents
    document_paths = ["sample_document.txt", "sample.txt"]
    documents = [preprocess_text(read_document(path)) for path in document_paths if read_document(path)]

    if not documents:
        print("No valid documents found. Exiting.")
        return

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    # Query example
    query = "What is artificial intelligence?"
    preprocessed_query = preprocess_text(query)
    print(f"\nPreprocessed Query: {preprocessed_query}")

    # Retrieve relevant document
    relevant_doc, similarity_score = retrieve_relevant_documents(preprocessed_query, documents, vectorizer)
    print(f"\nMost Relevant Document:\n{relevant_doc}")
    print(f"\nCosine Similarity Score: {similarity_score}")

    # Generate an answer using LLM
    answer = generate_answer_with_llm(query, relevant_doc)
    print(f"\nGenerated Answer:\n{answer}")

if __name__ == "__main__":
    main()
