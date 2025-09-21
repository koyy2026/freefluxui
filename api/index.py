from flask import Flask, jsonify, request
from openai import OpenAI
import os

app = Flask(__name__)

# 模型白名單
ALLOWED_MODELS = {
    "flux.1-schnell", "flux.1-dev", "flux.1-krea-dev",
    "flux.1.1-pro", "flux.1-kontext-pro"
}

# 新增：允許的圖像尺寸白名單
ALLOWED_SIZES = {
    "1024x1024",
    "1792x1024",
    "1024x1792"
}

@app.route('/api/generate', methods=['POST'])
def handle_image_generation():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

    data = request.get_json()
    user_prompt = data.get("prompt")
    model = data.get("model")
    size = data.get("size") # 獲取 size 參數

    if not all([user_prompt, model, size]):
        return jsonify({"error": "Prompt, model, and size are required"}), 400

    if model not in ALLOWED_MODELS:
        return jsonify({"error": "Invalid model specified"}), 400

    # 新增：驗證 size 參數
    if size not in ALLOWED_SIZES:
        return jsonify({"error": "Invalid size specified"}), 400

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.navy/v1")
        )

        image_response = client.images.generate(
            model=model,
            prompt=user_prompt,
            size=size, # 將 size 參數傳遞給 API
            n=1
        )

        image_url = image_response.data[0].url
        return jsonify({"imageUrl": image_url})

    except Exception as e:
        error_message = str(e)
        if hasattr(e, 'response') and e.response:
            try:
                error_data = e.response.json()
                error_message = error_data.get('error', {}).get('message', str(e))
            except:
                pass
        return jsonify({"error": error_message}), 500
