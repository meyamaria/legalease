import os
import json
import uuid
import datetime
import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import speech_recognition as sr
from gtts import gTTS
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import re

# Initialize Flask app
app = Flask(_name_)
CORS(app)

# Configure MongoDB
app.config["MONGO_URI"] = "mongodb://localhost:27017/legalease"
mongo = PyMongo(app)

# JWT Secret Key
app.config['SECRET_KEY'] = 'legalease_secret_key'

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create a simple model for outcome prediction
# In a production environment, this would be a pre-trained model loaded from a file
def create_dummy_model():
    # Sample data for training
    data = {
        'section': [302, 354, 420, 376, 302, 354, 420, 376],
        'duration_days': [120, 60, 90, 180, 150, 45, 75, 200],
        'evidence_strength': [0.8, 0.5, 0.6, 0.9, 0.7, 0.4, 0.5, 0.85],
        'prior_convictions': [1, 0, 2, 1, 0, 0, 1, 2],
        'outcome': [1, 0, 0, 1, 1, 0, 0, 1]  # 1: Guilty, 0: Not Guilty
    }
    df = pd.DataFrame(data)
    
    # Features and target
    X = df[['section', 'duration_days', 'evidence_strength', 'prior_convictions']]
    y = df['outcome']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Create or load the model
outcome_prediction_model = create_dummy_model()

# Sample legal database (would be in MongoDB in a full implementation)
legal_database = {
    "ipc_sections": {
        "302": {
            "title": "Punishment for murder",
            "description": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "max_punishment": "Death or life imprisonment"
        },
        "354": {
            "title": "Assault or criminal force to woman with intent to outrage her modesty",
            "description": "Whoever assaults or uses criminal force to any woman, intending to outrage or knowing it to be likely that he will thereby outrage her modesty, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
            "max_punishment": "2 years imprisonment"
        },
        "420": {
            "title": "Cheating and dishonestly inducing delivery of property",
            "description": "Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
            "max_punishment": "7 years imprisonment"
        }
    },
    "case_laws": [
        {
            "id": "1",
            "title": "State vs. John Doe",
            "ipc_section": "302",
            "summary": "The defendant was found guilty of murder and sentenced to life imprisonment.",
            "ruling": "Guilty"
        },
        {
            "id": "2",
            "title": "State vs. Jane Smith",
            "ipc_section": "354",
            "summary": "The defendant was acquitted due to lack of evidence.",
            "ruling": "Not Guilty"
        }
    ],
    "common_queries": {
        "what_are_my_rights_if_falsely_accused": {
            "english": "If you are falsely accused, you have the right to remain silent, the right to legal representation, and the right to a fair trial. It's advisable to consult with a lawyer immediately and not to make any statements without legal counsel present.",
            "malayalam": "തെറ്റായി ആരോപിക്കപ്പെട്ടാൽ, നിങ്ങൾക്ക് മൗനമായിരിക്കാനുള്ള അവകാശം, നിയമ പ്രാതിനിധ്യത്തിനുള്ള അവകാശം, നീതിയുക്തമായ വിചാരണയ്ക്കുള്ള അവകാശം എന്നിവയുണ്ട്. ഉടൻ അഭിഭാഷകനെ സമീപിക്കുകയും നിയമോപദേശം ഇല്ലാതെ പ്രസ്താവനകൾ നടത്താതിരിക്കുകയും ചെയ്യുന്നതാണ് നല്ലത്."
        },
        "how_to_file_police_complaint": {
            "english": "To file a police complaint, visit the nearest police station with your ID proof, write a detailed complaint including date, time, and location of the incident, and ensure you get a signed copy of the FIR for your records.",
            "malayalam": "പോലീസ് പരാതി നൽകാൻ, നിങ്ങളുടെ ID തെളിവുമായി ഏറ്റവും അടുത്തുള്ള പോലീസ് സ്റ്റേഷനിൽ സന്ദർശിക്കുക, സംഭവത്തിന്റെ തീയതി, സമയം, സ്ഥലം എന്നിവ ഉൾപ്പെടുന്ന വിശദമായ പരാതി എഴുതുക, കൂടാതെ നിങ്ങളുടെ രേഖകൾക്കായി FIR-ന്റെ ഒപ്പിട്ട പകർപ്പ് ലഭിക്കുന്നുണ്ടെന്ന് ഉറപ്പാക്കുക."
        }
    }
}

# Database helper functions
def json_serialize(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# Helper function to process natural language queries
def process_query(query, language="english"):
    # Tokenize the query
    tokens = word_tokenize(query.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Check for IPC section numbers
    ipc_sections = []
    for token in filtered_tokens:
        if token.isdigit() and token in legal_database["ipc_sections"]:
            ipc_sections.append(token)
    
    # Check for keywords
    keywords = {
        "rights": ["right", "rights", "entitled", "entitled to"],
        "complaint": ["complaint", "file", "report", "police", "fir"],
        "falsely accused": ["false", "falsely", "wrongly", "accused", "allegation"],
        "bail": ["bail", "release", "custody"],
        "theft": ["theft", "stolen", "rob", "robbery"],
        "assault": ["assault", "attack", "hit", "beat"],
        "cheating": ["cheat", "fraud", "deceive", "scam"]
    }
    
    matched_keywords = {}
    for category, category_keywords in keywords.items():
        for token in filtered_tokens:
            if token in category_keywords:
                matched_keywords[category] = matched_keywords.get(category, 0) + 1
    
    # Find best matching predefined query
    best_match = None
    best_match_score = 0
    
    for query_key, response in legal_database["common_queries"].items():
        score = 0
        query_keywords = query_key.split("_")
        for keyword in query_keywords:
            if keyword in filtered_tokens:
                score += 1
        
        if score > best_match_score:
            best_match_score = score
            best_match = query_key
    
    return {
        "ipc_sections": ipc_sections,
        "matched_keywords": matched_keywords,
        "best_match": best_match,
        "filtered_tokens": filtered_tokens
    }

# Function to generate response based on query analysis
def generate_response(query_analysis, language="english"):
    response = ""
    
    # If there's a direct match to a common query
    if query_analysis["best_match"] and query_analysis["best_match"] in legal_database["common_queries"]:
        return legal_database["common_queries"][query_analysis["best_match"]][language]
    
    # If IPC sections are mentioned
    if query_analysis["ipc_sections"]:
        section_info = []
        for section in query_analysis["ipc_sections"]:
            if section in legal_database["ipc_sections"]:
                info = legal_database["ipc_sections"][section]
                section_info.append(f"Section {section}: {info['title']} - {info['description']} Maximum punishment: {info['max_punishment']}")
        
        if section_info:
            if language == "english":
                response += "Here's information about the IPC sections you mentioned:\n\n"
                response += "\n\n".join(section_info)
            else:  # Malayalam
                response += "നിങ്ങൾ സൂചിപ്പിച്ച IPC വിഭാഗങ്ങളെക്കുറിച്ചുള്ള വിവരങ്ങൾ ഇതാ:\n\n"
                response += "\n\n".join(section_info)
    
    # If keywords about being falsely accused are present
    if "falsely accused" in query_analysis["matched_keywords"] and query_analysis["matched_keywords"]["falsely accused"] > 0:
        if language == "english":
            response += "\n\nIf you are falsely accused, you have the right to remain silent, the right to legal representation, and the right to a fair trial. It's advisable to consult with a lawyer immediately and not to make any statements without legal counsel present."
        else:  # Malayalam
            response += "\n\nതെറ്റായി ആരോപിക്കപ്പെട്ടാൽ, നിങ്ങൾക്ക് മൗനമായിരിക്കാനുള്ള അവകാശം, നിയമ പ്രാതിനിധ്യത്തിനുള്ള അവകാശം, നീതിയുക്തമായ വിചാരണയ്ക്കുള്ള അവകാശം എന്നിവയുണ്ട്. ഉടൻ അഭിഭാഷകനെ സമീപിക്കുകയും നിയമോപദേശം ഇല്ലാതെ പ്രസ്താവനകൾ നടത്താതിരിക്കുകയും ചെയ്യുന്നതാണ് നല്ലത്."
    
    # If keywords about filing a complaint are present
    if "complaint" in query_analysis["matched_keywords"] and query_analysis["matched_keywords"]["complaint"] > 0:
        if language == "english":
            response += "\n\nTo file a police complaint, visit the nearest police station with your ID proof, write a detailed complaint including date, time, and location of the incident, and ensure you get a signed copy of the FIR for your records."
        else:  # Malayalam
            response += "\n\nപോലീസ് പരാതി നൽകാൻ, നിങ്ങളുടെ ID തെളിവുമായി ഏറ്റവും അടുത്തുള്ള പോലീസ് സ്റ്റേഷനിൽ സന്ദർശിക്കുക, സംഭവത്തിന്റെ തീയതി, സമയം, സ്ഥലം എന്നിവ ഉൾപ്പെടുന്ന വിശദമായ പരാതി എഴുതുക, കൂടാതെ നിങ്ങളുടെ രേഖകൾക്കായി FIR-ന്റെ ഒപ്പിട്ട പകർപ്പ് ലഭിക്കുന്നുണ്ടെന്ന് ഉറപ്പാക്കുക."
    
    # Fallback response if no matching content found
    if not response:
        if language == "english":
            response = "I'm sorry, I couldn't find specific information about your query. Please try rephrasing your question or consult with a legal professional for personalized advice."
        else:  # Malayalam
            response = "ക്ഷമിക്കണം, നിങ്ങളുടെ ചോദ്യത്തെക്കുറിച്ച് പ്രത്യേക വിവരങ്ങൾ കണ്ടെത്താൻ എനിക്ക് കഴിഞ്ഞില്ല. ദയവായി നിങ്ങളുടെ ചോദ്യം പുനഃക്രമീകരിക്കാൻ ശ്രമിക്കുക അല്ലെങ്കിൽ വ്യക്തിഗത ഉപദേശത്തിനായി ഒരു നിയമ വിദഗ്ധനെ സമീപിക്കുക."
    
    return response

# Voice processing functions
def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"

def text_to_speech(text, language="en"):
    tts = gTTS(text=text, lang=language)
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    tts.save(temp_filename)
    return temp_filename

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Check if required fields are present
        required_fields = ['username', 'password', 'email']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Check if user already exists
        existing_user = mongo.db.users.find_one({"username": data['username']})
        if existing_user:
            return jsonify({"error": "Username already exists"}), 409
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create user document
        user = {
            "username": data['username'],
            "password": hashed_password,
            "email": data['email'],
            "created_at": datetime.datetime.utcnow(),
            "query_history": []
        }
        
        # Insert user into database
        user_id = mongo.db.users.insert_one(user).inserted_id
        
        return jsonify({"message": "User registered successfully", "user_id": str(user_id)}), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        # Check if required fields are present
        required_fields = ['username', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Find user
        user = mongo.db.users.find_one({"username": data['username']})
        
        # Verify user and password
        if not user or not check_password_hash(user['password'], data['password']):
            return jsonify({"error": "Invalid username or password"}), 401
        
        # Generate JWT token
        token_payload = {
            "user_id": str(user['_id']),
            "username": user['username'],
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }
        
        token = jwt.encode(token_payload, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": str(user['_id']),
                "username": user['username'],
                "email": user['email']
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Authentication middleware
def token_required(f):
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        
        try:
            # Decode token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = mongo.db.users.find_one({"_id": ObjectId(data['user_id'])})
            
            if not current_user:
                return jsonify({"error": "Invalid token"}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        
        return f(current_user, *args, **kwargs)
    
    decorated._name_ = f._name_
    return decorated

@app.route('/api/legal-query', methods=['POST'])
@token_required
def legal_query(current_user):
    try:
        data = request.get_json()
        
        # Check if required fields are present
        if 'query' not in data:
            return jsonify({"error": "Missing required field: query"}), 400
        
        language = data.get('language', 'english')
        if language not in ['english', 'malayalam']:
            return jsonify({"error": "Unsupported language. Supported languages: english, malayalam"}), 400
        
        # Process the query
        query_analysis = process_query(data['query'], language)
        
        # Generate response
        response = generate_response(query_analysis, language)
        
        # Save query to user history
        query_record = {
            "query": data['query'],
            "language": language,
            "response": response,
            "timestamp": datetime.datetime.utcnow()
        }
        
        mongo.db.users.update_one(
            {"_id": current_user['_id']},
            {"$push": {"query_history": query_record}}
        )
        
        return jsonify({
            "response": response,
            "analysis": query_analysis
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/voice-query', methods=['POST'])
@token_required
def voice_query(current_user):
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'english')
        
        if language not in ['english', 'malayalam']:
            return jsonify({"error": "Unsupported language. Supported languages: english, malayalam"}), 400
        
        # Save audio file temporarily
        temp_filename = f"temp_{uuid.uuid4()}.wav"
        audio_file.save(temp_filename)
        
        # Convert speech to text
        query_text = speech_to_text(temp_filename)
        
        # Remove temporary file
        os.remove(temp_filename)
        
        if query_text == "Could not understand audio" or query_text == "Could not request results":
            return jsonify({"error": query_text}), 400
        
        # Process the query
        query_analysis = process_query(query_text, language)
        
        # Generate response
        response_text = generate_response(query_analysis, language)
        
        # Convert response to speech
        speech_lang = "en" if language == "english" else "ml"
        response_audio_file = text_to_speech(response_text, speech_lang)
        
        # Save query to user history
        query_record = {
            "query": query_text,
            "language": language,
            "response": response_text,
            "query_type": "voice",
            "timestamp": datetime.datetime.utcnow()
        }
        
        mongo.db.users.update_one(
            {"_id": current_user['_id']},
            {"$push": {"query_history": query_record}}
        )
        
        # Return audio file path (in a real implementation, this would serve the file)
        return jsonify({
            "query_text": query_text,
            "response_text": response_text,
            "response_audio": f"/api/audio/{os.path.basename(response_audio_file)}",
            "analysis": query_analysis
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/outcome-prediction', methods=['POST'])
@token_required
def outcome_prediction(current_user):
    try:
        data = request.get_json()
        
        # Check if required fields are present
        required_fields = ['section', 'duration_days', 'evidence_strength', 'prior_convictions']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Prepare input for prediction
        input_data = {
            'section': [int(data['section'])],
            'duration_days': [int(data['duration_days'])],
            'evidence_strength': [float(data['evidence_strength'])],
            'prior_convictions': [int(data['prior_convictions'])]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = outcome_prediction_model.predict(input_df)
        prediction_proba = outcome_prediction_model.predict_proba(input_df)
        
        # Determine outcome
        outcome = "Likely Guilty" if prediction[0] == 1 else "Likely Not Guilty"
        confidence = round(max(prediction_proba[0]) * 100, 2)
        
        # Save prediction to user history
        prediction_record = {
            "case_details": data,
            "prediction": {
                "outcome": outcome,
                "confidence": confidence
            },
            "timestamp": datetime.datetime.utcnow()
        }
        
        mongo.db.users.update_one(
            {"_id": current_user['_id']},
            {"$push": {"prediction_history": prediction_record}}
        )
        
        return jsonify({
            "prediction": outcome,
            "confidence": confidence,
            "factors": {
                "section_impact": "High" if data['section'] in ['302', '376'] else "Medium",
                "evidence_impact": "Strong" if float(data['evidence_strength']) > 0.7 else "Moderate",
                "duration_impact": "Favorable" if int(data['duration_days']) < 90 else "Unfavorable",
                "prior_convictions_impact": "Negative" if int(data['prior_convictions']) > 0 else "Neutral"
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ipc-sections', methods=['GET'])
@token_required
def get_ipc_sections(current_user):
    try:
        # Return all IPC sections from the database
        return jsonify({"ipc_sections": legal_database["ipc_sections"]}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/case-laws', methods=['GET'])
@token_required
def get_case_laws(current_user):
    try:
        # Get query parameters
        section = request.args.get('section')
        
        # Filter case laws by section if provided
        if section:
            filtered_cases = [case for case in legal_database["case_laws"] if case["ipc_section"] == section]
            return jsonify({"case_laws": filtered_cases}), 200
        
        # Return all case laws
        return jsonify({"case_laws": legal_database["case_laws"]}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/history', methods=['GET'])
@token_required
def get_user_history(current_user):
    try:
        # Get query parameters
        query_type = request.args.get('type')
        limit = int(request.args.get('limit', 10))
        
        # Get user history
        user = mongo.db.users.find_one({"_id": current_user['_id']})
        
        history = {
            "query_history": user.get("query_history", [])[-limit:],
            "prediction_history": user.get("prediction_history", [])[-limit:]
        }
        
        # Filter by type if provided
        if query_type == "query":
            return jsonify({"history": history["query_history"]}), 200
        elif query_type == "prediction":
            return jsonify({"history": history["prediction_history"]}), 200
        
        return jsonify(history), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if _name_ == '_main_':
    app.run(debug=True)