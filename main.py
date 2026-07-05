import os
import smtplib
from email.message import EmailMessage
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "groq/compound")

# In-memory conversation store: username -> list of messages
CONVERSATIONS = {}
# Maximum number of messages to keep per conversation (includes system entry)
MAX_HISTORY_MESSAGES = 15


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "message": "Backend Groq siap menerima request."
    }), 200


@app.route("/api/ask", methods=["POST"])
def ask_ai():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "Pengguna")
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({
            "status": "error",
            "message": "Pesan AI tidak boleh kosong."
        }), 400

    # Ensure conversation exists for this username (preserve topic across requests)
    if username not in CONVERSATIONS:
        try:
            with open("Algoritma.txt", "r", encoding="utf-8") as f:
                algoritma = f.read()
        except Exception:
            algoritma = ""
        CONVERSATIONS[username] = [{"role": "system", "content": algoritma}]

    answer = generate_ai_response(username, message)

    return jsonify({
        "status": "success",
        "answer": answer,
        "input": {
            "username": username,
            "message": message
        }
    }), 200


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    if not username:
        return jsonify({"status": "error", "message": "username required"}), 400

    try:
        with open("Algoritma.txt", "r", encoding="utf-8") as f:
            algoritma = f.read()
    except Exception:
        algoritma = ""

    CONVERSATIONS[username] = [{"role": "system", "content": algoritma}]
    return jsonify({"status": "success", "message": "conversation reset"}), 200


def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY tidak ditemukan. Isi kunci Groq Anda di BackEnd/.env atau environment variable.")

    return Groq(api_key=api_key)


def is_invitation_message(text: str) -> bool:
    if not text:
        return False
    s = text.lower()
    keywords = [
        "kopdar", "ketemu", "undang", "undangan", "ajak", "ajakan", "besok", "kapan", "dateng", "datang", "gabung", "smala", "meetup", "temu", "ketemuan"
    ]
    return any(k in s for k in keywords)


def has_datetime_details(text: str) -> bool:
    """Return True if text contains explicit time or date details.

    This is a heuristic: looks for 'jam', 'pukul', weekdays, parts of day, 'tanggal', 'tgl', months, or numeric times.
    """
    if not text:
        return False
    s = text.lower()
    # words that usually indicate a clear time/date
    datetime_keywords = [
        "jam", "pukul", "tanggal", "tgl", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu",
        "sore", "pagi", "malam", "siang", "mei", "april", "maret", "juni", "juli", "agustus", "september", "oktober", "november", "desember",
        "besok pagi", "besok siang", "besok sore", "lusa", "minggu depan", "bulan depan"
    ]
    if any(k in s for k in datetime_keywords):
        # but 'besok' alone is ambiguous; ensure there's at least 'sore/pagi/siang' or 'jam' or weekday/month
        if "besok" in s and not any(x in s for x in ["jam", "pukul", "sore", "pagi", "siang", "malam"]):
            return False
        return True

    # numeric time like '3', '15:00', '3 pm' etc — look for digits near 'jam' or standalone hh:mm
    if re.search(r"\bjam\s*\d{1,2}\b", s) or re.search(r"\b\d{1,2}:\d{2}\b", s):
        return True

    return False


def send_notification_email(subject: str, body: str) -> bool:
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASS", "").strip()
    to_address = os.environ.get("NOTIFY_EMAIL_TO", "").strip()

    if not (smtp_user and smtp_pass and to_address):
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_user
    message["To"] = to_address
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(message)
        return True
    except Exception:
        return False


def trigger_invitation_email(username: str, message: str, ai_answer: str, summary: str = "") -> bool:
    subject = f"[Notifikasi] Undangan baru dari {username}"
    # Letakkan ringkasan AI di bagian atas supaya mudah dibaca oleh penerima
    body_parts = [
        "Halo Dafa,\n\n",
        f"Ringkasan undangan:\n{summary}\n\n" if summary else "",
        f"Dari: {username}\n\n",
        f"Pesan asli dari user:\n{message}\n\n",
        f"AI sudah merespon:\n{ai_answer}\n\n",
        "Silakan tindak lanjuti jika perlu.",
    ]

    body = "".join(body_parts)
    return send_notification_email(subject, body)


def summarize_invitation_text(message: str, ai_answer: str = None) -> str:
    """Return a short, explicit Indonesian explanation of the invitation.

    The summary should explain: 1) apakah ini ajakan/undangan, 2) kegiatan/tujuan (apa), 3) lokasi atau platform jika disebut (ke mana), 4) waktu yang disebut atau 'tidak disebutkan', dan 5) siapa pengundang jika jelas. Output 1-3 kalimat, nada netral.
    """
    try:
        client = get_groq_client()
        prompt_parts = [
            {
                "role": "system",
                "content": (
                    "Anda adalah asisten yang membuat PENJELASAN singkat (1-3 kalimat) tentang sebuah ajakan/undangan. "
                    "Jelaskan secara eksplisit: (a) apakah ini ajakan/undangan, (b) kegiatan atau tujuan (apa yang akan dilakukan), "
                    "(c) lokasi atau platform jika disebut (ke mana), (d) waktu yang disebut atau 'tidak disebutkan', dan (e) siapa pengundang jika jelas. "
                    "Jika suatu informasi tidak disebutkan dalam pesan asli, tuliskan 'tidak disebutkan' untuk bagian tersebut. "
                    "Tulis dalam bahasa Indonesia, nada netral dan ringkas."
                ),
            },
            {"role": "user", "content": f"Pesan pengguna:\n{message}"},
        ]
        if ai_answer:
            prompt_parts.append({"role": "user", "content": f"AI menjawab:\n{ai_answer}\nJika relevan, sertakan konteks singkat dari jawaban AI."})

        completion = client.chat.completions.create(
            messages=prompt_parts,
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            max_completion_tokens=160,
            temperature=0.15,
        )

        if getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            if getattr(first_choice, "message", None) is not None:
                return getattr(first_choice.message, "content", "").strip()

        return "(ringkasan gagal dibuat)"
    except Exception:
        return "(ringkasan gagal dibuat)"


def generate_ai_response(username: str, message: str) -> str:
    try:
        client = get_groq_client()

        # Ensure conversation exists (should be initialized in ask_ai)
        if username not in CONVERSATIONS:
            try:
                with open("Algoritma.txt", "r", encoding="utf-8") as f:
                    algoritma = f.read()
            except Exception:
                algoritma = ""
            CONVERSATIONS[username] = [{"role": "system", "content": algoritma}]

        # Append user message to conversation history
        user_entry = {"role": "user", "content": f"User: {username}\nMessage: {message}"}
        CONVERSATIONS[username].append(user_entry)

        # Build messages to send; start from stored history
        messages = list(CONVERSATIONS[username])

        # If invitation detected, add instruction
        if is_invitation_message(message):
            if has_datetime_details(message):
                invitation_instr = (
                    "Catatan: Pesan ini adalah ajakan/undangan dengan detail waktu/tanggal. "
                    "Jawab singkat konfirmasi dalam bahasa Indonesia (mis. 'Siap, saya ikut pada [waktu]'). "
                    "Sertakan tag [[SEND_EMAIL]] di akhir respons untuk memicu pengiriman undangan ke email jika relevan. "
                    "Jangan tanyakan lagi."
                )
            else:
                invitation_instr = (
                    "Catatan: Pesan ini tampak sebagai ajakan/undangan tetapi TIDAK ada detail waktu/tanggal yang jelas. "
                    "Tanyakan SATU pertanyaan singkat untuk klarifikasi (mis. 'Besok jam berapa?'). "
                    "Jangan mengirim email sampai user memberi detail lengkap."
                )
            messages.insert(1, {"role": "system", "content": invitation_instr})

        completion = client.chat.completions.create(
            messages=messages,
            model=os.environ.get("GROQ_MODEL", "groq/compound"),
            max_completion_tokens=512,
            temperature=0.7,
        )

        if getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            if getattr(first_choice, "message", None) is not None:
                content = getattr(first_choice.message, "content", str(completion)).strip()

                # If model requested sending email, create summary and email it (summary not shown to user)
                if "[[SEND_EMAIL]]" in content:
                    cleaned = content.replace("[[SEND_EMAIL]]", "").strip()
                    try:
                        summary = summarize_invitation_text(message, cleaned)
                    except Exception:
                        summary = "(ringkasan gagal dibuat)"

                    trigger_invitation_email(username, message, cleaned, summary)

                    # Save assistant reply to history and trim
                    CONVERSATIONS[username].append({"role": "assistant", "content": cleaned})
                    if len(CONVERSATIONS[username]) > MAX_HISTORY_MESSAGES:
                        CONVERSATIONS[username] = [CONVERSATIONS[username][0]] + CONVERSATIONS[username][- (MAX_HISTORY_MESSAGES - 1) :]

                    return cleaned

                # Normal assistant reply: save and return
                CONVERSATIONS[username].append({"role": "assistant", "content": content})
                if len(CONVERSATIONS[username]) > MAX_HISTORY_MESSAGES:
                    CONVERSATIONS[username] = [CONVERSATIONS[username][0]] + CONVERSATIONS[username][- (MAX_HISTORY_MESSAGES - 1) :]

                return content

        return str(completion)
    except Exception as exc:
        return f"Terjadi kesalahan saat memanggil Groq: {exc}"


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint tidak ditemukan."
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "status": "error",
        "message": "Terjadi kesalahan internal pada server."
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
