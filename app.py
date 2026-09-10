from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import traceback
import time

app = Flask(__name__)

# ============================================================
# DEBUG LOGGING
# ============================================================

def log(message):
    print(f"[DEBUG] {message}", flush=True)


# ============================================================
# CORS
# ============================================================

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)


# ============================================================
# REQUEST LOGGER
# ============================================================

@app.before_request
def log_request():
    log("=" * 60)
    log(f"REQUEST METHOD : {request.method}")
    log(f"REQUEST PATH   : {request.path}")
    log(f"REQUEST URL    : {request.url}")
    log(f"REMOTE ADDRESS : {request.remote_addr}")
    log(f"CONTENT TYPE   : {request.content_type}")
    log(f"USER AGENT     : {request.headers.get('User-Agent')}")
    log(f"ORIGIN         : {request.headers.get('Origin')}")

    if request.method in ["POST", "OPTIONS"]:
        log(f"ACCESS CONTROL REQUEST METHOD : "
            f"{request.headers.get('Access-Control-Request-Method')}")
        log(f"ACCESS CONTROL REQUEST HEADERS: "
            f"{request.headers.get('Access-Control-Request-Headers')}")


@app.after_request
def log_response(response):
    log(f"RESPONSE STATUS: {response.status_code}")
    log(f"RESPONSE TYPE  : {response.content_type}")
    log("=" * 60)

    return response


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

log("Starting LearnAI backend...")
log(f"Python version: {os.sys.version}")
log(f"Working directory: {os.getcwd()}")
log(f"Model: {MODEL_NAME}")

try:
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    log("MODEL LOADED SUCCESSFULLY")

except Exception as e:
    log("!!! MODEL LOADING FAILED !!!")
    log(str(e))
    traceback.print_exc()

    tokenizer = None
    model = None


# ============================================================
# SYSTEM PROMPT
# ============================================================

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


# ============================================================
# ROOT / HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():
    log("ROOT ENDPOINT HIT")

    return jsonify({
        "status": "online",
        "model": MODEL_NAME,
        "generate_endpoint": "/generate",
        "model_loaded": model is not None
    })


# ============================================================
# GENERATE
# ============================================================

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():

    log("GENERATE ENDPOINT HIT")
    log(f"Method = {request.method}")

    # --------------------------------------------------------
    # CORS PREFLIGHT
    # --------------------------------------------------------

    if request.method == "OPTIONS":
        log("CORS PREFLIGHT REQUEST DETECTED")
        log("Returning 204")

        response = jsonify({
            "status": "cors_preflight_ok"
        })

        response.status_code = 204
        return response


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if model is None or tokenizer is None:
        log("ERROR: MODEL IS NOT LOADED")

        return jsonify({
            "error": "Model failed to load",
            "debug": "Check backend logs"
        }), 500


    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    try:
        data = request.get_json(silent=True)

        log(f"JSON DATA: {data}")

    except Exception as e:
        log("FAILED TO READ JSON")
        log(str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Invalid JSON"
        }), 400


    if not data:
        log("ERROR: REQUEST BODY IS EMPTY")

        return jsonify({
            "error": "Request body is empty"
        }), 400


    if "message" not in data:
        log("ERROR: 'message' FIELD IS MISSING")

        return jsonify({
            "error": "Missing 'message' field",
            "received_fields": list(data.keys())
        }), 400


    user_message = str(data["message"])

    log(f"USER MESSAGE LENGTH: {len(user_message)}")
    log(f"USER MESSAGE: {user_message[:500]}")


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = f"""
{SYSTEM_PROMPT}

USER:
{user_message}

ASSISTANT:
"""

    log("Prompt created")
    log(f"Prompt length: {len(prompt)}")


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    try:
        log("Tokenizing...")

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        log(
            f"Input tokens: "
            f"{inputs['input_ids'].shape[1]}"
        )

    except Exception as e:
        log("TOKENIZATION FAILED")
        log(str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Tokenization failed",
            "debug": str(e)
        }), 500


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    try:
        start_time = time.time()

        log("STARTING MODEL GENERATION")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generation_time = time.time() - start_time

        log(
            f"MODEL GENERATION FINISHED "
            f"({generation_time:.2f}s)"
        )

    except Exception as e:
        log("MODEL GENERATION FAILED")
        log(str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Model generation failed",
            "debug": str(e)
        }), 500


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    try:
        generated = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        log(f"ANSWER LENGTH: {len(answer)}")
        log(f"ANSWER: {answer[:500]}")

    except Exception as e:
        log("DECODING FAILED")
        log(str(e))
        traceback.print_exc()

        return jsonify({
            "error": "Failed to decode model output",
            "debug": str(e)
        }), 500


    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    log("RETURNING SUCCESS RESPONSE")

    return jsonify({
        "response": answer,
        "debug": {
            "model": MODEL_NAME,
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generation_time_seconds": round(generation_time, 2)
        }
    })


# ============================================================
# 404 DEBUG HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):
    log("!!! 404 - ENDPOINT NOT FOUND !!!")
    log(f"Requested path: {request.path}")
    log(f"Requested method: {request.method}")

    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "method": request.method,
        "available_routes": [
            "/",
            "/generate"
        ]
    }), 404


# ============================================================
# 405 DEBUG HANDLER
# ============================================================

@app.errorhandler(405)
def method_not_allowed(error):
    log("!!! 405 - METHOD NOT ALLOWED !!!")
    log(f"Requested path: {request.path}")
    log(f"Requested method: {request.method}")

    allowed = []

    for rule in app.url_map.iter_rules():
        if request.path == rule.rule:
            allowed.append({
                "path": rule.rule,
                "methods": sorted(rule.methods)
            })

    log(f"Allowed methods for this path: {allowed}")

    return jsonify({
        "error": "Method not allowed",
        "path": request.path,
        "method_used": request.method,
        "allowed_methods": allowed
    }), 405


# ============================================================
# 500 DEBUG HANDLER
# ============================================================

@app.errorhandler(500)
def internal_error(error):
    log("!!! 500 INTERNAL SERVER ERROR !!!")
    traceback.print_exc()

    return jsonify({
        "error": "Internal server error",
        "debug": str(error)
    }), 500


# ============================================================
# SHOW ROUTES
# ============================================================

log("REGISTERED ROUTES:")

for rule in app.url_map.iter_rules():
    log(f"{rule} -> {sorted(rule.methods)}")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    log("=" * 60)
    log("SERVER STARTING")
    log(f"HOST: 0.0.0.0")
    log(f"PORT: {port}")
    log("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )