from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from chat_general import get_response
import json

app = Flask(__name__)
def get_dataset():
    with open("intents.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    intents = data.get("intents", [])
    patterns = sum(len(intent.get("patterns", [])) for intent in intents)
    responses = sum(len(intent.get("responses", [])) for intent in intents)
    return intents, patterns, responses    

@app.route("/admin_app")
def admin_app():
    intents, patterns, responses = get_dataset()
    return render_template(
        "admin_app.html",
        intents=intents,
        patterns=patterns,
        responses=responses
    )