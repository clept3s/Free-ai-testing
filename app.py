from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded.")


SYSTEM_PROMPT = """
You are LearnAI's private recommendation engine.

Determine how a specific person should learn something based
on the information they provide.

Consider their:
- learning goal
- current experience
- available time
- preferred learning methods
- difficulties
- motivation
- constraints

Choose appropriate techniques such as active recall,
spaced repetition, practice exercises, project-based learning,
worked examples, flashcards, quizzes, reading, videos,
short focused sessions, progressive difficulty and feedback.

Do not blindly recommend every technique.

Return:

LEARNING APPROACH
WHY
PLAN
FIRST STEP

Keep the answer practical and personalized.
Do not invent information about the user.
"""


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "model": MODEL_NAME
    })


@app.route("/generate", methods=["POST"])
def generate():

    data = request.get_json(silent=True)

    if not data or "message" not in data:
        return jsonify({
            "error": "Missing message"
        }), 400

    user_message = str(data["message"])

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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.05
    )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return jsonify({
        "response": answer
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000
    )