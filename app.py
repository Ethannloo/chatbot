import os
import json
from typing import List, Optional, Dict

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__, static_folder='.', static_url_path='')  
CORS(app)  


def load_knowledge(file_path: str) -> Dict:
    """
    Loads or creates a default knowledge base JSON file.
    """
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        default_data = {"questions": []}
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(default_data, file, indent=2)
        return default_data
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_knowledge(file_path: str, data: Dict) -> None:
    """
    Saves the knowledge base dictionary to the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def encode_questions(model: SentenceTransformer, questions: List[str]):
    return model.encode(questions, convert_to_tensor=True)

def find_best_match(user_question: str, questions: List[str], model: SentenceTransformer) -> Optional[str]:
    if not questions:
        return None
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    question_embeddings = encode_questions(model, questions)
    similarities = util.cos_sim(user_embedding, question_embeddings)
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[0][best_match_idx].item()
    # Return the question only if it’s above a similarity threshold
    if best_match_score > 0.7:
        return questions[best_match_idx]
    return None

def get_answer(question: str, knowledge_base: Dict) -> Optional[str]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
    return None


# Load the pre-trained SentenceTransformer model at startup
model = SentenceTransformer('all-MiniLM-L6-v2')


knowledge_base = load_knowledge('knowledge_base.json')


@app.route('/')
def serve_index():
    """
    Serve the index.html file so the user can load the chatbot UI
    at the root URL (http://127.0.0.1:5000).
    """
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_input = data.get('question', '')

    if user_input.lower() == "quit":
        return jsonify({"response": "Goodbye!", "needs_teaching": False})

    all_questions = [q["question"] for q in knowledge_base["questions"]]
    best_match = find_best_match(user_input, all_questions, model)

    if best_match:
        answer = get_answer(best_match, knowledge_base)
        return jsonify({"response": answer or "Sorry, no answer found.", "needs_teaching": False})
    else:
        return jsonify({
            "response": "I don't understand. Can you teach me?",
            "needs_teaching": True
        })

@app.route('/teach', methods=['POST'])
def teach():
    data = request.get_json(force=True)
    user_question = data.get('question', '')
    user_answer = data.get('answer', '')

    knowledge_base['questions'].append({'question': user_question, 'answer': user_answer})
    save_knowledge('knowledge_base.json', knowledge_base)
    return jsonify({"response": "Got it. Thanks for teaching me!"})

if __name__ == '__main__':
    # Adjust the working directory if necessary
    if '__file__' in globals():
        os.chdir(os.path.dirname(__file__))

    # Run the Flask app
    app.run(debug=True)
