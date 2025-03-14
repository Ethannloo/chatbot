import os
import json
from typing import List, Optional, Dict

from sentence_transformers import SentenceTransformer, util


def load_knowledge(file_path: str) -> Dict:
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        # Create a default knowledge base if the file is missing or empty
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
    
    # Encode the user's question
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    # Encode the knowledge base questions
    question_embeddings = encode_questions(model, questions)
    # Compute similarity scores
    similarities = util.cos_sim(user_embedding, question_embeddings)
    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[0][best_match_idx].item()
    # Only return the match if the similarity score is above a threshold
    if best_match_score > 0.7:
        return questions[best_match_idx]
    return None


def get_answer(question: str, knowledge_base: Dict) -> Optional[str]:
    for q in knowledge_base["questions"]:
        if q["question"] == question:
            return q["answer"]
    return None


def chat_bot() -> None:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    try:
        knowledge_base = load_knowledge('knowledge_base.json')
    except FileNotFoundError as e:
        print(e)
        return

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("Bot: Goodbye!")
            break

        all_questions = [q["question"] for q in knowledge_base["questions"]]
        best_match = find_best_match(user_input, all_questions, model)

        if best_match:
            answer = get_answer(best_match, knowledge_base)
            if answer:
                print(f'Bot: {answer}')
            else:
                print("Bot: I don't understand (no matching answer found).")
        else:
            print("Bot: I don't understand. Can you teach me?")
            new_answer = input("Type the answer or 'Skip' to skip: ")

            if new_answer.lower() != 'skip':
                knowledge_base['questions'].append({'question': user_input, 'answer': new_answer})
                save_knowledge('knowledge_base.json', knowledge_base)
                print('Bot: Thank you for teaching me the new response.')


if __name__ == '__main__':
    if '__file__' in globals():
        os.chdir(os.path.dirname(__file__))

    chat_bot()
