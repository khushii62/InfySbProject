import os
import shutil
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cohere
# Initialize Cohere client
COHERE_API_KEY = "teqwH8AqLu8MXqSQhor28E6yx6X363HfXBPOPfvo"  # Replace with your actual API key
co = cohere.Client(COHERE_API_KEY)

# Load product data
product_data_path = r"D:/Users/productdata.csv"
product_data = pd.read_csv(product_data_path)

# Google Sheets setup
def authenticate_google_sheets():
    try:
        creds = Credentials.from_service_account_file(
            r'D:/Users/ccuser/Downloads/nifty-buffer-445714-g5-21f65091c5a9.json',
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key('1hITPd5x7jJ12gsX3H9pLbDTHPH65ltd_aTHC4H6cAN8').sheet1
        return sheet
    except Exception as e:
        print(f"Error in Google Sheets authentication: {e}")
        return None

# Function to detect sentiment using VADER
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

# Function to handle objections using Cohere
def handle_objection(query):
    try:
        response = co.generate(
            model="command-xlarge",  # Choose an appropriate model
            prompt=f"Handle the following objection or question: {query}",
            max_tokens=100,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error in objection handling: {e}")
        return "Sorry, I couldn't generate a response."

# Function to recommend similar products
def recommend_product(query):
    try:
        # Ensure ProductName column exists and query is lowercased
        if 'ProductName' not in product_data.columns:
            print("ProductName column missing in the data.")
            return [{"message": "Product data is invalid or missing required columns."}]
        
        query = query.lower()
        product_data['ProductName'] = product_data['ProductName'].str.lower()

        # Find products similar to the user query
        similar_products = product_data[product_data['ProductName'].str.contains(query, na=False)]

        if not similar_products.empty:
            recommendations = similar_products[['ProductName', 'category', 'price']].to_dict(orient='records')
            return recommendations
        else:
            return [{"message": "No similar products found."}]
    except Exception as e:
        print(f"Error in product recommendation: {e}")
        return [{"message": "Error recommending products."}]


# Real-time speech recognition, sentiment analysis, objection handling, and product recommendation
# Real-time speech recognition with feedback
# Real-time speech recognition with feedback and sentiment analysis
def real_time_analysis():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Say 'stop' to stop the process.")
    try:
        sheet = authenticate_google_sheets()
        while True:
            with mic as source:
                print("Listening...")  # Added feedback for listening
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                print("Recognizing...")  # Added feedback for recognizing
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")
                
                if 'stop' in text.lower():
                    print("Stopping real-time analysis...")
                    break  # Exit the loop and stop the process

                # Check if the recognized text is likely incomplete (cut off mid-sentence)
                if len(text.split()) < 3:  # Assuming 3 words is a threshold for incompleteness
                    print("Sorry, I didn't catch that. Could you please repeat your question?")
                    continue  # Skip processing and prompt user again
                
                # Perform sentiment analysis
                sentiment = analyze_sentiment(text)
                print(f"Sentiment: {sentiment}")
                
                # Perform product search
                search_product(text)
                
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
if __name__ == "_main_":
    real_time_analysis()