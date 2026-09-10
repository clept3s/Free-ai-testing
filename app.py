from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==========================================
# CONFIG
# ==========================================

MODEL_NAME = "Qwen/Qwen3-0.6B"


# ==========================================
# FLASK
# ==========================================

app = Flask(__name__)

CORS(app)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


print("Loading Qwen3 model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)


print("Model loaded successfully.")


# ==========================================
# SYSTEM PROMPT
# ==========================================

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


# ==========================================
# GENERATE
# ==========================================

@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json()

    if not data or "message" not in data:

        return jsonify({
            "error": "Missing message"
        }), 400


    user_message = data["message"]


    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },

        {
            "role": "user",
            "content": user_message
        }

    ]


    # Convert chat messages into the model's format

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)


    # Generate

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=400,

            temperature=0.7,

            do_sample=True,

            repetition_penalty=1.05
        )


    # Remove the original prompt

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return jsonify({
        "response": response
    })


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "status": "online",
        "model": MODEL_NAME
    })


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )
