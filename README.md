# Data-Mining-Agent

This project is an AI-powered **Document Mining Agent** that retrieves relevant documents based on a user query and generates AI-driven responses using OpenAI's GPT model.

## **Features**
- Reads and processes text documents.
- Uses **TF-IDF and Cosine Similarity** to find relevant documents.
- Integrates **OpenAI GPT model** (`text-davinci-003`) for generating answers.
- Implements **text preprocessing** (lemmatization, stopword removal, tokenization).

## **Setup & Installation**
### **1. Install Dependencies**
Ensure you have Python installed, then run:
```sh
pip install openai numpy pandas scikit-learn nltk
