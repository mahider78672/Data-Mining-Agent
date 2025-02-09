import openai  # For LLM integration (make sure you've installed openai package)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize OpenAI API key (replace with your key)
# openai.api_key = 'use the key here'

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the stemmer
stemmer = PorterStemmer()

# Function to read document
def read_document(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        print(f"Document content from {file_path} loaded successfully!")
        return content
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file path.")
        return None

# Preprocessing function (including stemming)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in cleaned_tokens]
    return ' '.join(stemmed_tokens)

# Function to retrieve relevant documents based on cosine similarity
def retrieve_relevant_documents(query, documents, vectorizer):
    query_vector = vectorizer.transform([query])
    tfidf_matrix = vectorizer.transform(documents)
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    relevant_doc_index = np.argmax(cosine_similarities)  # Get the most similar document
    return documents[relevant_doc_index], cosine_similarities[0][relevant_doc_index]

# Function to use LLM (GPT) for text generation based on the retrieved document
def generate_answer_with_llm(query, relevant_doc):
    prompt = f"Question: {query}\n\nRelevant Document: {relevant_doc}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",  # GPT model (you can switch to a different model if needed)
        prompt=prompt,
        max_tokens=100,  # Adjust based on desired response length
        temperature=0.7  # Control creativity
    )
    answer = response.choices[0].text.strip()
    return answer

# Main function
def main():
    print("Document Mining Agent Initialized!")

    # Read documents
    documents = []
    document_paths = ["sample_document.txt", "sample.txt"]
    for path in document_paths:
        content = read_document(path)
        if content:
            processed_content = preprocess_text(content)
            documents.append(processed_content)

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
