import os
import time
import logging
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --------------------------------------------------------------
# CONFIGURATION (change only here to upgrade model)
# --------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 500
# --------------------------------------------------------------

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   # allow all origins for simplicity

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load model once at startup
model = None
tokenizer = None
model_loaded = False

def load_model():
    global model, tokenizer, model_loaded
    logging.info("Loading model %s ...", MODEL_NAME)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model_loaded = True
    logging.info("Model loaded in %.2f sec", time.time() - start)

load_model()

# Private system prompt (kept server‑side)
SYSTEM_PROMPT = """You are LearnAI's private recommendation engine.

Your task is to determine how a specific person should learn something based on the information they provide.

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
Choose only techniques that actually fit the individual.

The recommendation should be practical rather than generic.
If information is missing, work with what is available instead of inventing facts.

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

def build_prompt(user_message: str) -> str:
    return f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{user_message}"

@app.route("/", methods=["GET"])
ndef health():
    return jsonify({
        "status": "online",
        "model": MODEL_NAME,
        "model_loaded": model_loaded
    })

@app.route("/generate", methods=["POST", "OPTIONS"])
ndef generate():
    # CORS preflight handling
    if request.method == "OPTIONS":
        logging.info("CORS preflight received for /generate")
        resp = app.make_default_options_response()
        return resp

    # Log request details
    logging.info("POST /generate from %s", request.headers.get("Origin"))
    logging.info("Content-Type: %s", request.headers.get("Content-Type"))

    if not request.is_json:
        logging.warning("Invalid request: not JSON")
        abort(400, description="Invalid JSON")

    data = request.get_json()
    logging.info("Received JSON: %s", data)

    user_msg = data.get("message", "").strip()
    if not user_msg:
        logging.warning("Empty message")
        abort(400, description="Missing 'message' field")

    logging.info("Message length: %d", len(user_msg))

    # Tokenization & generation
    try:
        prompt = build_prompt(user_msg)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        logging.info("Tokenization done, %d tokens", inputs["input_ids"].shape[1])

        start_gen = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_time = time.time() - start_gen
        logging.info("Generation completed in %.2f sec", gen_time)

        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove the prompt part from the output
        response_text = response_text[len(prompt):].strip()

        return jsonify({"response": response_text})
    except Exception as e:
        logging.exception("Generation error")
        abort(500, description=str(e))

# Custom error handlers
@app.errorhandler(404)
def not_found(e):
    logging.warning("404: %s", request.path)
    return jsonify({"error": "Not found", "path": request.path}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    allowed = ["OPTIONS", "POST"]
    logging.warning("405 on %s, method %s", request.path, request.method)
    return jsonify({
        "error": "Method not allowed",
        "path": request.path,
        "method_used": request.method,
        "allowed_methods": allowed
    }), 405

@app.errorhandler(500)
def internal_error(e):
    logging.error("500 error: %s", e)
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)