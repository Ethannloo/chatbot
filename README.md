Overview
This project is a minimalistic, Q&A-style chatbot that uses semantic similarity to answer user queries based on a local knowledge base. It relies on the SentenceTransformer library (using the all-MiniLM-L6-v2 model) to encode both user questions and stored knowledge entries into vector embeddings. When a user asks a question, the chatbot computes the similarity between the user’s query and existing questions in its knowledge base (a JSON file). If the similarity score exceeds a certain threshold (e.g., 0.7), the best-matching answer is returned immediately.
If the user’s question does not match any stored entry with enough confidence, the chatbot indicates that it doesn’t understand and prompts the user to provide an answer. Once the user supplies an answer, the bot automatically appends this new Q/A pair to the JSON knowledge base, so it can respond correctly in the future. Over time, the chatbot “learns” from these interactions without needing a retraining step—its storage just keeps growing.
You can interact with the chatbot in two primary ways:
Command-Line Interface (CLI):
Run main.py to chat directly in a terminal. If the bot is unable to answer a question, it will ask you to teach it right in the console.
Web Frontend:
A simple Flask server app.py and an HTML/JS interface index.html provide a minimal “GPT-style” web chat. Users can type queries, receive instant responses, and teach the bot if needed. Once the new answer is saved, the bot can respond correctly to that query in the future.
This design makes the chatbot both light and extensible:
Lightweight: It stores data in a simple JSON file (knowledge_base.json) and uses a single transformer model for semantic search.
Easily Extensible: You can change thresholds, refine how the bot matches questions, or expand the frontend/UI without overhauling the entire codebase.
Dynamic: No complex retraining or database overhead—new entries are appended on the fly, which fosters quick experimentation and easy version control.
