from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer, util
import json
import os


# Import your chatbot functions
from main import load_knowledge, save_knowledge, find_best_match, get_answer

app = Flask(__name__)

# Initialize chatbot model and knowledge base
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base = load_knowledge('knowledge_base.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({'response': 'Please enter a valid message.'})

    # Find the best match in the knowledge base
    best_match = find_best_match(user_input, [q['question'] for q in knowledge_base['questions']], model)
    if best_match:
        answer = get_answer(best_match, knowledge_base)
        return jsonify({'response': answer})
    else:
        return jsonify({'response': "I don't understand. Can you teach me?"})

if __name__ == '__main__':
    app.run(debug=True)
