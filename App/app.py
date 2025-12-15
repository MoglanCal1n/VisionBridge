from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io, base64
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./moondream_crosswalks_finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.eval()
print(f"✓ Model loaded on {DEVICE}")

def ask_image(image, question):
    """Pune o întrebare despre o imagine - folosind inferență manuală."""
    with torch.no_grad():
        enc_image = model.encode_image(image)
    
    prompt = f"\n\nQuestion: {question}\n\nAnswer:"
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    
    text_emb = model.text_model.get_input_embeddings()(tokens)
    
    inputs_embeds = torch.cat([enc_image, text_emb], dim=1)
    
    max_new_tokens = 100
    generated_tokens = tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model.text_model(
                inputs_embeds=inputs_embeds,
                use_cache=False
            )
            logits = outputs.logits
            
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            
            next_token_emb = model.text_model.get_input_embeddings()(next_token)
            inputs_embeds = torch.cat([inputs_embeds, next_token_emb], dim=1)
    
    answer_tokens = generated_tokens[0, tokens.shape[1]:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    return answer.strip()

@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        data = request.get_json()
        image_data = data.get("image")
        question = data.get("question", "Is there a crosswalk in this image? If so, what's the color of the semaphor if there is any?")

        if not image_data:
            return jsonify({"error": "No image received"}), 400

        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        answer = ask_image(image, question)

        return jsonify({
            "answer": answer,
            "question": question
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint pentru întrebări custom."""
    try:
        data = request.get_json()
        image_data = data.get("image")
        question = data.get("question")

        if not image_data or not question:
            return jsonify({"error": "Image and question required"}), 400

        image_bytes = base64.b64decode(image_data.split(",")[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        answer = ask_image(image, question)

        return jsonify({
            "answer": answer,
            "question": question
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Check dacă serverul funcționează."""
    return jsonify({
        "status": "healthy",
        "model": MODEL_PATH,
        "device": DEVICE
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=13000)