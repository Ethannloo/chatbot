from sentence_transformers import SentenceTransformer, util
import json
import os


def load_knowledge(file_path: str) -> dict:
   if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
       # Create a default knowledge base if the file is missing or empty
       default_data = {"questions": []}
       with open(file_path, "w") as file:
           json.dump(default_data, file, indent=2)
       return default_data
   with open(file_path, "r") as file:
       return json.load(file)


def save_knowledge(file_path: str, data: dict):
   with open(file_path, "w") as file:
       json.dump(data, file, indent=2)


def encode_questions(model, questions: list[str]) -> list:
   return model.encode(questions, convert_to_tensor=True)


def find_best_match(user_question: str, questions: list[str], model) -> str | None:
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


def get_answer(question: str, knowledge_base: dict) -> str | None:
   for q in knowledge_base["questions"]:
       if q["question"] == question:
           return q["answer"]


def chat_bot():
   # Load the pre-trained SentenceTransformer model
   model = SentenceTransformer('all-MiniLM-L6-v2')
   try:
       knowledge_base: dict = load_knowledge('knowledge_base.json')
   except FileNotFoundError as e:
       print(e)
       return


   while True:
       user_input: str = input("You: ")


       if user_input.lower() == "quit":
           break


       best_match: str | None = find_best_match(
           user_input, [q["question"] for q in knowledge_base["questions"]], model
       )


       if best_match:
           answer: str = get_answer(best_match, knowledge_base)
           print(f'Bot: {answer}')
       else:
           print('Bot: I don\'t understand. Can you teach me?')
           new_answer: str = input("Type the answer or 'Skip' to Skip:")


           if new_answer.lower() != 'skip':
               knowledge_base['questions'].append({'question': user_input, 'answer': new_answer})
               save_knowledge('knowledge_base.json', knowledge_base)
               print('Bot: Thank you for teaching me the new response.')


if __name__ == '__main__':
   os.chdir(os.path.dirname(__file__))  # Ensure the working directory is the script's location
   chat_bot()
