from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json
import os

app = Flask(__name__)

# Load or create knowledge base
def load_knowledge(file_path: str) -> dict:
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        default_data = {"questions": []}
        with open(file_path, "w") as file:
            json.dump(default_data, file, indent=2)
        return default_data
    with open(file_path, "r") as file:
        return json.load(file)

def save_knowledge(file_path: str, data: dict):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

def encode_questions(model, questions: list[str]):
    return model.encode(questions, convert_to_tensor=True)

def find_best_match(user_question: str, questions: list[str], model):
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

def get_answer(question: str, knowledge_base: dict):
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]

# Initialize model and knowledge base at startup
model = SentenceTransformer('all-MiniLM-L6-v2')
knowledge_base = load_knowledge('knowledge_base.json')

@app.route('/chat', methods=['POST'])
def chat():
    """Accepts JSON of the form: {"question": "some text"}"""
    data = request.get_json(force=True)
    user_input = data.get('question', '')

    # If user enters "quit" – optional logic:
    if user_input.lower() == "quit":
        return jsonify({"response": "Goodbye!"})

    best_match = find_best_match(
        user_input, [q["question"] for q in knowledge_base["questions"]], model
    )

    if best_match:
        answer = get_answer(best_match, knowledge_base)
        return jsonify({"response": answer})
    else:
        # If no match is found, the bot doesn’t automatically add new Q/A
        # but instructs the front end to show a "teach me" prompt if desired
        return jsonify({"response": "I don't understand. Can you teach me?", "needs_teaching": True})

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
    # Run on http://127.0.0.1:5000 by default
    app.run(debug=True)
