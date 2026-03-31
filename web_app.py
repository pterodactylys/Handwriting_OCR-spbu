"""Flask web app for OCR testing."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

from data.preprocessing import ImageConfig, preprocess_image
from data.vocab import CharVocab
from models.crnn import CRNN
from models.ctc_decoder import greedy_decode

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model = None
vocab = None
image_cfg = None
device = torch.device("cpu")
checkpoint_path = None
manifest_lookup = {}


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/load-checkpoint", methods=["POST"])
def load_checkpoint():
    global model, vocab, image_cfg, checkpoint_path
    
    data = request.json
    ckpt_path = data.get("checkpoint_path")
    
    if not ckpt_path:
        return jsonify({"error": "No path provided"}), 400
    
    try:
        path = Path(ckpt_path).expanduser().resolve()
        if not path.exists():
            return jsonify({"error": f"File not found: {path}"}), 404
        
        ckpt = torch.load(path, map_location=device)
        vocab = CharVocab.load(ckpt["vocab_path"])
        image_cfg = ImageConfig(height=int(ckpt["image_height"]), width=int(ckpt["image_width"]))
        
        model_obj = CRNN(num_classes=vocab.size).to(device)
        model_obj.load_state_dict(ckpt["model_state_dict"])
        model_obj.eval()
        model = model_obj
        
        checkpoint_path = path
        return jsonify({"status": "success", "message": f"Loaded: {path.name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-manifest", methods=["POST"])
def load_manifest():
    global manifest_lookup
    
    data = request.json
    manifest_path = data.get("manifest_path")
    
    if not manifest_path:
        return jsonify({"error": "No path provided"}), 400
    
    try:
        path = Path(manifest_path).expanduser().resolve()
        if not path.exists():
            return jsonify({"error": f"File not found: {path}"}), 404
        
        lookup = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                lookup[str(Path(row["image_path"]).resolve())] = row["text"]
        
        manifest_lookup = lookup
        return jsonify({"status": "success", "message": f"Loaded: {len(lookup)} samples"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or vocab is None or image_cfg is None:
        return jsonify({"error": "Checkpoint not loaded"}), 400
    
    try:
        if "file" in request.files:
            file = request.files["file"]
            image_data = Image.open(BytesIO(file.read())).convert("L")
            image_path = None
        else:
            image_path = request.json.get("image_path")
            if not image_path:
                return jsonify({"error": "No image provided"}), 400
            path = Path(image_path).expanduser().resolve()
            if not path.exists():
                return jsonify({"error": f"File not found: {path}"}), 404
            image_data = Image.open(path).convert("L")
        
        tensor = preprocess_image(image_data, image_cfg).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            log_probs = logits.log_softmax(dim=2)
            pred_ids = greedy_decode(log_probs, blank_idx=vocab.blank_idx)[0]
        
        pred_text = vocab.decode(pred_ids)
        
        gt_text = ""
        if image_path:
            key = str(Path(image_path).expanduser().resolve())
            gt_text = manifest_lookup.get(key, "")
        
        return jsonify({
            "prediction": pred_text if pred_text else "<empty>",
            "ground_truth": gt_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Handwriting OCR</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #333; margin-bottom: 30px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        label { display: block; font-weight: 600; margin-bottom: 8px; color: #333; }
        input[type="text"], input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        input[type="file"] { cursor: pointer; }
        button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 600; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .status { padding: 10px; border-radius: 4px; margin-top: 10px; display: none; }
        .status.success { background: #d4edda; color: #155724; }
        .status.error { background: #f8d7da; color: #721c24; }
        .image-preview { max-width: 100%; max-height: 300px; border-radius: 4px; margin: 10px 0; }
        .text-area { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-family: monospace; resize: none; }
        .section-title { font-size: 14px; font-weight: 600; color: #666; margin-top: 15px; margin-bottom: 10px; }
        .button-group { display: flex; gap: 10px; margin-top: 10px; }
        .button-group button { flex: 1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖋️ Handwriting OCR</h1>
        
        <div class="card">
            <h2 style="font-size: 18px; margin-bottom: 15px;">Setup</h2>
            <label>Checkpoint path:</label>
            <input type="text" id="ckptPath" placeholder="e.g., artifacts/checkpoints/crnn_small_v2.pt"
                   value="artifacts/checkpoints/crnn_small_v2.pt">
            <button onclick="loadCheckpoint()">Load Checkpoint</button>
            <div id="ckptStatus" class="status"></div>
            
            <label style="margin-top: 15px;">Manifest path (optional):</label>
            <input type="text" id="manifestPath" placeholder="e.g., artifacts/data_small/test.jsonl"
                   value="artifacts/data_small/test.jsonl">
            <button onclick="loadManifest()">Load Manifest</button>
            <div id="manifestStatus" class="status"></div>
        </div>
        
        <div class="card">
            <h2 style="font-size: 18px; margin-bottom: 15px;">Test Image</h2>
            <label>Upload or paste path:</label>
            <input type="file" id="imageFile" accept="image/*">
            <input type="text" id="imagePath" placeholder="Or paste image path here" style="margin-top: 10px;">
            <div class="button-group">
                <button onclick="predictFromFile()">Predict from File</button>
                <button onclick="predictFromPath()">Predict from Path</button>
            </div>
            <div id="predStatus" class="status"></div>
            
            <img id="preview" class="image-preview" style="display: none;">
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="section-title">Prediction</div>
                <textarea class="text-area" id="prediction" rows="4" readonly></textarea>
            </div>
            <div class="card">
                <div class="section-title">Ground Truth (from manifest)</div>
                <textarea class="text-area" id="groundTruth" rows="4" readonly></textarea>
            </div>
        </div>
    </div>
    
    <script>
        function showStatus(elementId, message, isError = false) {
            const el = document.getElementById(elementId);
            el.textContent = message;
            el.className = "status " + (isError ? "error" : "success");
            el.style.display = "block";
        }
        
        function loadCheckpoint() {
            const path = document.getElementById("ckptPath").value.trim();
            if (!path) {
                showStatus("ckptStatus", "Please provide a path", true);
                return;
            }
            
            fetch("/api/load-checkpoint", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ checkpoint_path: path })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showStatus("ckptStatus", "Error: " + data.error, true);
                } else {
                    showStatus("ckptStatus", data.message);
                }
            })
            .catch(e => showStatus("ckptStatus", "Request failed: " + e, true));
        }
        
        function loadManifest() {
            const path = document.getElementById("manifestPath").value.trim();
            if (!path) {
                showStatus("manifestStatus", "Please provide a path", true);
                return;
            }
            
            fetch("/api/load-manifest", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ manifest_path: path })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showStatus("manifestStatus", "Error: " + data.error, true);
                } else {
                    showStatus("manifestStatus", data.message);
                }
            })
            .catch(e => showStatus("manifestStatus", "Request failed: " + e, true));
        }
        
        function predictFromFile() {
            const file = document.getElementById("imageFile").files[0];
            if (!file) {
                showStatus("predStatus", "Select an image first", true);
                return;
            }
            
            const formData = new FormData();
            formData.append("file", file);
            
            // Show preview
            const reader = new FileReader();
            reader.onload = e => {
                document.getElementById("preview").src = e.target.result;
                document.getElementById("preview").style.display = "block";
            };
            reader.readAsDataURL(file);
            
            fetch("/api/predict", {
                method: "POST",
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showStatus("predStatus", "Error: " + data.error, true);
                } else {
                    document.getElementById("prediction").value = data.prediction;
                    document.getElementById("groundTruth").value = data.ground_truth || "(not in manifest)";
                    showStatus("predStatus", "Prediction complete");
                }
            })
            .catch(e => showStatus("predStatus", "Request failed: " + e, true));
        }
        
        function predictFromPath() {
            const path = document.getElementById("imagePath").value.trim();
            if (!path) {
                showStatus("predStatus", "Enter image path first", true);
                return;
            }
            
            fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_path: path })
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showStatus("predStatus", "Error: " + data.error, true);
                } else {
                    document.getElementById("prediction").value = data.prediction;
                    document.getElementById("groundTruth").value = data.ground_truth || "(not in manifest)";
                    showStatus("predStatus", "Prediction complete");
                }
            })
            .catch(e => showStatus("predStatus", "Request failed: " + e, true));
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting OCR Web App on http://localhost:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
