# Q&A Chatbot
This project is a lightweight chatbot that answers questions using semantic similarity with a local knowledge base. It's designed to be minimal, easy to use, and capable of learning from user input without needing retraining.

# Features
Answers user queries by comparing them to stored questions using a transformer model (all-MiniLM-L6-v2)

Stores question-answer pairs in a local JSON file

Learns new answers dynamically when it doesn’t recognize a question

Two ways to interact: through the command line or a simple web interface

No database or retraining needed just updates the JSON file as it learns

# How it Works
The user asks a question.

The chatbot encodes the question and compares it to existing entries in the knowledge base.

If it finds a similar enough question (above a set threshold), it returns the matching answer.

If no match is found, it asks the user to provide the answer.

The new question and answer are added to the knowledge base for future use.

# Interfaces
Command Line Interface (CLI):
Run the chatbot directly in the terminal. If it can't answer a question, it will ask you to teach it right there.

Web Interface:
A minimal web frontend using Flask and basic HTML/JavaScript. Type your questions in the browser and get instant responses. You can also teach the bot when it doesn't know the answer.

# Design Goals
Lightweight: Uses a simple JSON file for storage and a single transformer model for matching.

Extensible: Easy to adjust matching thresholds, improve logic, or build a better UI.

Dynamic: Learns on the fly from user input—no need to retrain or manage a database.
