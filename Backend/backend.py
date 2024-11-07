from flask import Flask, request, jsonify
from transformers import pipeline, Conversation

app = Flask(__name__)

conversation_pipeline = pipeline("conversational", model="facebook/blenderbot-400M-distill")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Convert the user message to a Conversation instance
    conversation = Conversation(user_message)

    # Use the LLM to generate a response
    response = conversation_pipeline(conversation)
    generated_response = response.generated_responses[0]

    return jsonify({"response": generated_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
