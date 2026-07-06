import os
import smtplib
import time
import json
import traceback
import urllib.request
import urllib.error
from email.message import EmailMessage
import re

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
# Batas token yang boleh dibuat model per jawaban (bisa diatur lewat .env tanpa ubah kode)
MAX_COMPLETION_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "200"))

# In-memory conversation store: username -> list of messages
CONVERSATIONS = {}
# Timestamp (time.time()) pesan terakhir yang masuk per username
LAST_ACTIVITY = {}
# Maximum number of messages to keep per conversation (includes system entry)
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "8"))
# Kalau user diam lebih lama dari ini (detik), conversation-nya otomatis di-reset
# saat dia kirim pesan lagi. Ini juga yang menjaga jumlah token per request tetap
# kecil karena history lama/basi tidak ikut menumpuk dan terkirim terus.
INACTIVITY_RESET_SECONDS = int(os.environ.get("INACTIVITY_RESET_SECONDS", "60"))

# Cache isi Algoritma.txt di memori supaya tidak buka file setiap kali ada
# conversation baru/reset.
_SYSTEM_PROMPT_CACHE = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is None:
        try:
            with open("Algoritma.txt", "r", encoding="utf-8") as f:
                _SYSTEM_PROMPT_CACHE = f.read()
        except Exception:
            _SYSTEM_PROMPT_CACHE = ""
    return _SYSTEM_PROMPT_CACHE


def _new_conversation() -> list:
    return [{"role": "system", "content": _load_system_prompt()}]


def _ensure_active_conversation(username: str) -> None:
    """Pastikan username sudah punya conversation. Kalau user sudah tidak
    kirim pesan selama lebih dari INACTIVITY_RESET_SECONDS, conversation lama
    dibuang dan mulai lagi dari system prompt (konteks fresh + hemat token)."""
    now = time.time()
    last_seen = LAST_ACTIVITY.get(username)
    if username not in CONVERSATIONS or (last_seen is not None and now - last_seen > INACTIVITY_RESET_SECONDS):
        CONVERSATIONS[username] = _new_conversation()
    LAST_ACTIVITY[username] = now


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

    # Pastikan conversation ada; otomatis reset kalau user idle > 1 menit
    _ensure_active_conversation(username)

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

    CONVERSATIONS[username] = _new_conversation()
    LAST_ACTIVITY[username] = time.time()
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

    # numeric time like '3', '15:00', '3 pm' etc â€” look for digits near 'jam' or standalone hh:mm
    if re.search(r"\bjam\s*\d{1,2}\b", s) or re.search(r"\b\d{1,2}:\d{2}\b", s):
        return True

    return False


def send_notification_email(subject: str, body: str) -> bool:
    """Kirim email notifikasi.

    Railway (plan Free/Trial/Hobby) memblokir SEMUA outbound SMTP (port 25,
    465, 587, 2525) untuk mencegah abuse. Ini yang bikin smtplib selalu gagal
    connect di server walau kredensial & kode sudah benar — kalau dites dari
    laptop/lokal tetap jalan normal karena ISP rumah tidak memblokir port itu,
    jadi kelihatannya seperti "bug" padahal itu firewall Railway.

    Fix: kirim lewat Resend (https://resend.com), yang pakai HTTPS API biasa
    (port 443) — port ini tidak diblokir platform manapun, termasuk semua
    plan Railway. Kalau RESEND_API_KEY belum di-set di .env, otomatis fallback
    ke SMTP biasa (berguna kalau nanti pindah host lain yang tidak memblokir
    SMTP, atau upgrade ke Railway plan Pro).
    """
    resend_api_key = os.environ.get("RESEND_API_KEY", "").strip()
    if resend_api_key:
        return _send_via_resend(subject, body, resend_api_key)
    return _send_via_smtp(subject, body)


def _send_via_resend(subject: str, body: str, api_key: str) -> bool:
    to_address = os.environ.get("NOTIFY_EMAIL_TO", "").strip()
    # "from" harus pakai domain yang sudah diverifikasi di Resend. Selama belum
    # verifikasi domain sendiri, "onboarding@resend.dev" cuma bisa kirim ke
    # alamat email akun Resend kamu sendiri — cukup untuk notifikasi internal
    # yang tujuannya memang cuma ke NOTIFY_EMAIL_TO.
    from_address = os.environ.get("RESEND_FROM", "onboarding@resend.dev").strip()

    if not to_address:
        return False

    payload = json.dumps({
        "from": from_address,
        "to": [to_address],
        "subject": subject,
        "text": body,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare (di depan api.resend.com) memblokir default User-Agent
            # bawaan urllib ("Python-urllib/3.x") sebagai bot -> 403 error code: 1010.
            # Header custom di bawah ini yang menghindari blokir tsb.
            "User-Agent": "Mozilla/5.0 (compatible; FexieBackend/1.0)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode(errors="ignore")
        except Exception:
            detail = ""
        print("Resend HTTPError:", e.code, detail)
        return False
    except Exception as e:
        traceback.print_exc()
        print("Resend exception:", repr(e))
        return False


def _send_via_smtp(subject: str, body: str) -> bool:
    smtp_host = "smtp.gmail.com"
    smtp_port = 465
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
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(message)
        return True

    except Exception as e:
        traceback.print_exc()
        print(repr(e))
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
    """Ringkasan singkat undangan untuk isi email notifikasi internal.

    Versi lama memanggil Groq LAGI di sini (1 request + token tambahan) padahal
    hasilnya cuma masuk ke badan email dan tidak pernah dibaca user. Versi ini
    menyusun ringkasan langsung dari heuristik yang sudah ada di file ini
    (has_datetime_details), jadi 0 request tambahan ke Groq.
    """
    waktu = "disebutkan dalam pesan" if has_datetime_details(message) else "tidak disebutkan secara eksplisit"
    lines = [
        "Pesan terdeteksi sebagai ajakan/undangan.",
        f"Detail waktu: {waktu}.",
        f"Isi pesan asli: {message.strip()}",
    ]
    if ai_answer:
        lines.append(f"Respon AI ke user: {ai_answer.strip()}")
    return " ".join(lines)


def generate_ai_response(username: str, message: str) -> str:
    try:
        client = get_groq_client()

        # Safety net kalau fungsi ini dipanggil tanpa lewat /api/ask dulu;
        # sekaligus menerapkan aturan auto-reset saat idle > 1 menit.
        _ensure_active_conversation(username)

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
            model=os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.5,
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