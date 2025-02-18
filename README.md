Chatbot with Flask and Semantic Search
Overview
This project is a simple chatbot application built using Flask and Sentence Transformers for natural language processing. It uses semantic similarity search to match user questions with pre-defined answers stored in a JSON knowledge base. The console version can also learn new answers from user input.
Key Features
Flask Web Application with REST API for chatbot interaction.
Semantic Search powered by Sentence Transformers (MiniLM-L6-v2) to understand the meaning behind user queries.
Knowledge Base (JSON) stores questions and answers for easy updates and persistence.
Console Chatbot can learn and expand its knowledge base by saving new questions and answers.
Web Interface allows users to chat with the bot in a browser.
How It Works
User Message: The user sends a message through the web interface or console.
Backend Processing:
Flask receives the message and passes it to the chatbot logic.
The message is compared to stored questions using semantic similarity.
If a close match is found, the corresponding answer is returned.
If no match is found (console version only), the bot asks the user to teach it the correct response.
Response Display: The chatbot returns the response to the user.

Installation and Setup
Install dependencies:
pip install flask sentence-transformers

Run the Flask app:
python app.py

Open the chatbot in your browser:
http://127.0.0.1:5000/

Using the Console Version
To run the console chatbot instead:
python main.py

Type your message, and if the bot doesn't know the answer, you can teach it a new response. Type quit to exit.
Future Improvements
Add the learning feature to the web version.
Store knowledge base in a database for scalability.
Explore more advanced NLP models for better understanding.
