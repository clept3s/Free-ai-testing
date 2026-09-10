from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Allow the GitHub Pages frontend to communicate with the backend
CORS(
    app,
    resources={
        r"/generate": {
            "origins": "*"
        }
    },
    methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

print("Model loaded!")


SYSTEM_PROMPT = """
You are LearnAI's private recommendation engine.

Your task is to determine how a specific person should learn
something based on the information they provide.

Analyze:
1. Their learning goal
2. Their current experience
3. Their available time
4. Their preferred learning methods
5. Their difficulties
6. Their motivation
7. Their environment and constraints

Select appropriate learning techniques from:
- Active recall
- Spaced repetition
- Practice exercises
- Project-based learning
- Worked examples
- Flashcards
- Quizzes
- Reading
- Videos
- Short focused sessions
- Progressive difficulty
- Frequent feedback

Do not blindly recommend every technique.
Choose the techniques that best fit the individual.

The recommendation should be practical rather than generic.

If important information is missing, work with what is available
rather than inventing facts.

Return:

LEARNING APPROACH
Explain the recommended overall approach.

WHY
Explain why it fits this person.

PLAN
Give a practical starting plan.

FIRST STEP
Give the user one concrete thing they can do first.

Keep the response clear and useful.
"""


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model": MODEL_NAME
    })


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():

    # Handle browser CORS preflight
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True)

    if not data or "message" not in data:
        return jsonify({
            "error": "Missing message"
        }), 400

    user_message = data["message"]

    prompt = f"""
{SYSTEM_PROMPT}

USER:
{user_message}

ASSISTANT:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return jsonify({
        "response": answer
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port
    )