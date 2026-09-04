from urllib import response
import os
import signal
from datetime import datetime
import json
import time
import random
import re
import pickle
import subprocess
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from chat_oss import get_response, load_model
import threading
import json

intent_name = "intents_oss_generated.json"

STATUS_FILE = "training_status.json"
LOG_FILE = "training.log"

app = Flask(__name__)
app.secret_key = "chatbot_dpmptsp_secret_key"

training_process = None
training_status = {
    "running": False,
    "message": "Training belum dijalankan",
    "logs": []
}

def get_dataset_info():
    with open(intent_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_intents = len(data["intents"])
    total_patterns = sum(len(intent["patterns"]) for intent in data["intents"])

    return total_intents, total_patterns

def parse_responses(text):
    """
    Mengubah isi textarea menjadi list responses.

    Aturan:
    - Baris kosong = pemisah antar response
    - Baris biasa = bagian dari response
    - Baris yang diawali '-' = item list
    - Enter biasa tetap menjadi newline dalam response
    """

    lines = text.splitlines()

    responses = []
    current_response = []

    for line in lines:

        # Pertahankan isi baris, tetapi hilangkan
        # spasi di bagian awal dan akhir
        line = line.strip()

        # ==========================================
        # BARIS KOSONG
        # ==========================================

        if not line:

            # Jika sedang ada response,
            # simpan sebagai response baru
            if current_response:

                responses.append(
                    "\n".join(current_response)
                )

                current_response = []

            continue

        # ==========================================
        # BARIS BIASA ATAU ITEM LIST
        # ==========================================

        current_response.append(line)

    # ==========================================
    # RESPONSE TERAKHIR
    # ==========================================

    if current_response:

        responses.append(
            "\n".join(current_response)
        )

    return responses

def get_dataset():
    with open(intent_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    intents = data["intents"]
    return intents

@app.route("/")
def home():
    total_intents, total_patterns = get_dataset_info()
    return render_template(
        "base_new.html",
        total_intents=total_intents,
        total_patterns=total_patterns
    )

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json["message"]
    result = get_response(message)
    return jsonify({
        "response": result["response"],
        "intent": result["intent"],
        "confidence": result["confidence"]
    })

@app.route("/admin_app", methods=["GET"])
def admin_app():
    intents = get_dataset()
    return render_template(
        "admin/admin_app.html",
        intents=intents
    )

@app.route("/log", methods=["GET"])
def log_training():
    return render_template("admin/log_training.html")

@app.route("/admin/train", methods=["POST"])
def start_training():
    global training_process

    status = get_training_status()

    # Cek apakah training masih berjalan
    if status["running"]:

        return jsonify({
            "success": False,
            "message": "Training sedang berlangsung."
        }), 409

    try:

        # Kosongkan log lama
        with open(
            LOG_FILE,
            "w",
            encoding="utf-8"
        ) as log:

            log.write(
                "TRAINING_STARTED\n"
            )

        # Jalankan train.py sebagai proses terpisah
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "train.py"
            ],
            stdout=open(
                LOG_FILE,
                "a",
                encoding="utf-8"
            ),
            stderr=subprocess.STDOUT,

            # Proses tidak bergantung pada request browser
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt"
            else 0
        )

        training_process = process

        status = {
            "running": True,
            "status": "training",
            "message": "Training sedang berlangsung...",
            "pid": process.pid,
            "started_at": datetime.now().isoformat(),
            "finished_at": None
        }

        save_training_status(status)

        threading.Thread(
            target=monitor_training,
            args=(process,),
            daemon=True
        ).start()

        return jsonify({
            "success": True,
            "message": "Training berhasil dimulai.",
            "pid": process.pid
        })

    except Exception as e:

        status = {
            "running": False,
            "status": "error",
            "message": str(e),
            "pid": None,
            "started_at": None,
            "finished_at": datetime.now().isoformat()
        }

        save_training_status(status)

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/admin/train/cancel", methods=["POST"])
def cancel_training():
    global training_process

    status = get_training_status()
    if not status.get("running"):
        return jsonify({
            "success": False,
            "message": "Tidak ada training yang sedang berjalan."
        }), 409

    process = training_process
    if process is None:
        pid = status.get("pid")
        if pid is None:
            return jsonify({
                "success": False,
                "message": "PID training tidak ditemukan."
            }), 404
        try:
            if os.name == "nt":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    else:
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass

        try:
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    training_process = None

    cancel_status = {
        "running": False,
        "status": "cancelled",
        "message": "Training dibatalkan oleh admin.",
        "pid": None,
        "started_at": status.get("started_at"),
        "finished_at": datetime.now().isoformat()
    }
    save_training_status(cancel_status)

    return jsonify({
        "success": True,
        "message": "Training berhasil dibatalkan."
    })

def get_training_status():
    if not os.path.exists(STATUS_FILE):
        return {
            "running": False,
            "status": "idle",
            "message": "Training belum dijalankan.",
            "pid": None,
            "started_at": None,
            "finished_at": None
        }
    try:
        with open(
            STATUS_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return {
            "running": False,
            "status": "error",
            "message": "Status training tidak dapat dibaca.",
            "pid": None,
            "started_at": None,
            "finished_at": None
        }
    
def save_training_status(status):
    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            status,
            f,
            indent=4,
            ensure_ascii=False
        )

def monitor_training(process):
    global training_process

    process.wait()
    status = get_training_status()

    if status.get("status") == "cancelled":
        training_process = None
        return

    training_process = None
    status["running"] = False
    if process.returncode == 0:
        status["status"] = "completed"
        status["message"] = (
            "Training selesai. Model terbaru tersedia."
        )
    else:
        status["status"] = "failed"
        status["message"] = (
            "Training gagal. Periksa training.log."
        )
    status["finished_at"] = (
        datetime.now().isoformat()
    )
    save_training_status(status)

@app.route("/admin/train/status")
def training_status():
    status = get_training_status()
    return jsonify(status)

@app.route("/admin/train/log")
def training_log():
    if not os.path.exists(LOG_FILE):
        return jsonify({
            "logs": []
        })
    try:
        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            logs = f.readlines()
        return jsonify({
            "logs": logs
        })
    except Exception as e:
        return jsonify({
            "logs": [],
            "error": str(e)
        }), 500

@app.route("/admin/add", methods=["GET", "POST"])
def add_intent():
    with open(intent_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    if request.method == "POST":
        # ==========================================
        # TAG
        # ==========================================
        tag = request.form.get("tag", "").strip()
        # ==========================================
        # PATTERNS
        # ==========================================
        patterns_text = request.form.get(
            "patterns",
            ""
        )
        patterns = [
            p.strip()
            for p in patterns_text.splitlines()
            if p.strip()
        ]
        # ==========================================
        # RESPONSE
        # ==========================================
        response_text = request.form.get(
            "responses",
            ""
        )
        responses = parse_responses(
            response_text
        )
        # ==========================================
        # VALIDASI TAG
        # ==========================================
        if not tag:
            return render_template(
                "admin/add_intent.html",
                error="Tag intent wajib diisi.",
                tag=tag,
                patterns=patterns_text,
                responses=response_text
            )
        # ==========================================
        # CEK TAG DUPLIKAT
        # ==========================================
        existing_tags = [
            intent["tag"]
            for intent in data["intents"]
        ]
        if tag in existing_tags:
            return render_template(
                "admin/add_intent.html",
                error="Tag intent sudah digunakan.",
                tag=tag,
                patterns=patterns_text,
                responses=response_text
            )
        # ==========================================
        # VALIDASI PATTERN
        # ==========================================
        if not patterns:
            return render_template(
                "admin/add_intent.html",
                error="Minimal harus ada satu pattern.",
                tag=tag,
                patterns=patterns_text,
                responses=response_text
            )
        # ==========================================
        # VALIDASI RESPONSE
        # ==========================================
        if not responses:
            return render_template(
                "admin/add_intent.html",
                error="Minimal harus ada satu response.",
                tag=tag,
                patterns=patterns_text,
                responses=response_text
            )
        # ==========================================
        # BUAT INTENT
        # ==========================================
        new_intent = {
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        }
        data["intents"].append(
            new_intent
        )
        # ==========================================
        # SIMPAN JSON
        # ==========================================
        with open(
            intent_name,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )
        return redirect(
            url_for("admin_app")
        )
    return render_template(
        "admin/add_intent.html"
    )

@app.route(
    "/admin/edit/<tag>",
    methods=["GET", "POST"]
)
def edit_intent(tag):
    with open(
        intent_name,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)
    # ==========================================
    # CARI INTENT
    # ==========================================
    intent = next(
        (
            i
            for i in data["intents"]
            if i["tag"] == tag
        ),
        None
    )
    if intent is None:
        return "Intent tidak ditemukan", 404
    # ==========================================
    # POST
    # ==========================================
    if request.method == "POST":
        # ======================================
        # PATTERNS
        # ======================================
        patterns_text = request.form.get(
            "patterns",
            ""
        )
        intent["patterns"] = [
            p.strip()
            for p in patterns_text.splitlines()
            if p.strip()
        ]
        # ======================================
        # RESPONSES
        # ======================================
        response_text = request.form.get(
            "responses",
            ""
        )
        responses = parse_responses(
            response_text
        )
        # ======================================
        # VALIDASI
        # ======================================
        if not intent["patterns"]:
            return render_template(
                "admin/edit_intent.html",
                intent=intent,
                responses=response_text,
                error="Minimal harus ada satu pattern."
            )
        if not responses:
            return render_template(
                "admin/edit_intent.html",
                intent=intent,
                responses=response_text,
                error="Minimal harus ada satu response."
            )
        # ======================================
        # UPDATE RESPONSE
        # ======================================
        intent["responses"] = responses
        # ======================================
        # SIMPAN
        # ======================================
        with open(
            intent_name,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=4
            )
        return redirect(
            url_for("admin_app")
        )
    # ==========================================
    # GET
    # ==========================================
    # Gabungkan kembali responses agar dapat
    # diedit di textarea.
    #
    # Dua response dipisahkan oleh satu
    # baris kosong.
    responses_text = "\n\n".join(
        intent.get("responses", [])
    )
    patterns_text = "\n".join(
        intent.get("patterns", [])
    )
    return render_template(
        "admin/edit_intent.html",
        intent=intent,
        patterns=patterns_text,
        responses=responses_text
    )

@app.route("/admin/delete/<tag>", methods=["POST"])
def delete_intent(tag):
    with open(intent_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["intents"] = [
        i for i in data["intents"]
        if i["tag"] != tag
    ]
    with open(intent_name, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )
    return redirect(url_for("admin_app"))

if __name__ == "__main__":
    app.run(
        debug=True
    )