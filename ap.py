import pandas as pd 
import speech_recognition as sr
from flask import Flask, request, jsonify
from flask_cors import CORS
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cohere
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


# Initialize Cohere client
COHERE_API_KEY = ""
co = cohere.Client(COHERE_API_KEY)

# Initialize Elasticsearch client
es = Elasticsearch("http://localhost:9200")
index_name = "productdata"

# Load product data
product_data_path = r"D:/Users/productdata.csv"
product_data = pd.read_csv(product_data_path)
product_data = product_data.fillna('')
# Load objection data
objection_questions_path = r"D:/milestone3/objections_questions.csv"
objection_questions = pd.read_csv(objection_questions_path).fillna('')

#Google Sheets
def authenticate_google_sheets():
    try:
        creds = Credentials.from_service_account_file(
            r'D:/Users/ccuser/Downloads/nifty-buffer-445714-g5-21f65091c5a9.json',
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key('').sheet1
        return sheet
    except Exception as e:
        print(f"Error in Google Sheets authentication: {e}")
        return None

# Sentiment analysis using VADER
def analyze_sentiment(text):
    try:
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(text)
        if sentiment_scores['compound'] > 0.05:
            return "Positive"
        elif sentiment_scores['compound'] < -0.05:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Error analyzing sentiment"

# Load objection questions dataset
def handle_objection(query,):
    try:
        response = co.generate(
            model="command-xlarge",
            prompt=f"Handle the following objection or question: {query}",
            max_tokens=100,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error in objection handling: {e}")
        return "Sorry, I couldn't generate a response."

# Generate embedding using Cohere
def generate_embedding(text):
    try:
        response = co.embed(texts=[text], model="embed-english-v2.0")
        return response.embeddings[0]
    except Exception as e:
        print(f"Error generating embedding for text: {text}\n{e}")
        return None

# Elasticsearch query for product recommendations
def search_product(query):
    try:
        query_embedding = generate_embedding(query)
        if query_embedding is None:
            print("Failed to generate embedding for query.")
            return

        # Product search query to Elasticsearch
        search_query = {
            "size": 5,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        response = es.search(index=index_name, body=search_query)

        hits = response['hits']['hits']
        
        if hits:
            # If matching products found
            top_product = hits[0]["_source"]
            print(f"Product: {top_product['name']}, Price: {top_product['price']}, Category: {top_product['category']}, Description: {top_product['description']}")
            
            # Recommend similar products in the same category
            category = top_product['category']
            similar_products_query = {
                "size": 4,
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"category": category}}
                        ]
                    }
                }
            }
            similar_products_response = es.search(index=index_name, body=similar_products_query)
            similar_hits = similar_products_response['hits']['hits']
            
            print(f"Similar products in the {category} category:")
            for hit in similar_hits:
                source = hit["_source"]
                print(f"Name: {source['name']}, Price: {source['price']}")
        else:
            print("No matching products found.")
        
        objection_response = handle_objection(query)
        print(f"Objection Response: {objection_response}")
        
    except Exception as e:
        print(f"Error during search: {e}")


# Real-time speech recognition with feedback
def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Say 'stop' to stop the process.")
    try:
        sheet = authenticate_google_sheets()
        while True:
            with mic as source:
                print("Listening...") 
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                print("Recognizing...")  
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")
                
                if 'stop' in text.lower():
                    print("Stopping real-time analysis...")
                    break  # Exit the loop and stop the process
                
                # Call the product search function
                search_product(text)

                # Sentiment analysis
                sentiment = analyze_sentiment(text)
                print(f"Sentiment: {sentiment}")
                
                # Log the results in Google Sheets
                if sheet:
                    sheet.append_row([text, sentiment, "Product or Objection handled."])
                    print("Logged to Google Sheets successfully.")
                else:
                    print("Google Sheets logging skipped due to authentication error.")
                
            except sr.UnknownValueError:
                print("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Error with the Speech Recognition service: {e}")
            except Exception as e:
                print(f"Error during processing: {e}")

    except Exception as e:
        print(f"Error in real-time analysis: {e}")


# Start real-time analysis
if __name__ == "__main__":
    real_time_analysis()
