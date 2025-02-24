import os
import json
from typing import List, Optional, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS  # Optional, if you need cross-origin access
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)  # Enable this if your front-end is on a different origin (localhost vs 127.0.0.1 etc.)

def load_knowledge(file_path: str) -> Dict:
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        default_data = {"questions": []}
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(default_data, file, indent=2)
        return default_data
    
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_knowledge(file_path: str, data: Dict) -> None:
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
    if best_match_score > 0.7:
        return questions[best_match_idx]
    return None

def get_answer(question: str, knowledge_base: Dict) -> Optional[str]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
    return None

# Initialize model and knowledge base at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base = load_knowledge('knowledge_base.json')

@app.route('/chat', methods=['POST'])
def chat():
    """Accepts JSON of the form: {"question": "some text"}"""
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
    """Accepts JSON of the form: {"question": "some text", "answer": "some text"}"""
    data = request.get_json(force=True)
    user_question = data.get('question', '')
    user_answer = data.get('answer', '')

    knowledge_base['questions'].append({'question': user_question, 'answer': user_answer})
    save_knowledge('knowledge_base.json', knowledge_base)
    return jsonify({"response": "Got it. Thanks for teaching me!"})

if __name__ == '__main__':
    # If running in a standard environment with __file__ defined
    if '__file__' in globals():
        os.chdir(os.path.dirname(__file__))

    # Flask will run on http://127.0.0.1:5000 by default
    app.run(debug=True)
