"""Flask app: Login -> Upload -> Processing -> Result -> Performance.
Every page except login/register needs a logged-in user."""
import hmac
import json
import os
import re
import secrets
import time
import uuid
from functools import wraps
from pathlib import Path

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_from_directory, session, url_for)

from src import auth, config, preprocessing
from src.predict import ModelNotTrainedError, NailScreeningModel

config.INSTANCE_DIR.mkdir(exist_ok=True)
config.UPLOAD_DIR.mkdir(exist_ok=True)
auth.init_db()


def _secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    if not config.SECRET_KEY_PATH.exists():          # generated once, kept between restarts
        config.SECRET_KEY_PATH.write_text(secrets.token_hex(32))
    return config.SECRET_KEY_PATH.read_text().strip()


app = Flask(__name__)
app.config.update(
    SECRET_KEY=_secret_key(),
    MAX_CONTENT_LENGTH=(config.MAX_UPLOAD_MB + 1) * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

UID_RE = re.compile(r"^[0-9a-f]{32}$")
MEDIA_RE = re.compile(r"^([0-9a-f]{32})_(original|resized|gray|enhanced|roi|denoised|cutout|skin_mask|nail_only|nail_enhanced|edges)\.jpg$")
_model = None


# ---------------------------------------------------------------- helpers
def get_model():
    global _model
    if _model is None:
        _model = NailScreeningModel()
    return _model


def model_ready():
    try:
        get_model()
        return True
    except ModelNotTrainedError:
        return False


def login_required(view):
    @wraps(view)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            flash("Please login to continue.", "warning")
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapper


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]


@app.before_request
def check_csrf():
    if request.method == "POST":
        sent = request.form.get("csrf_token", "")
        if not hmac.compare_digest(sent, session.get("csrf", "")):
            abort(400)


@app.after_request
def no_cache(resp):
    if request.endpoint != "static":
        resp.headers["Cache-Control"] = "no-store"   # back button must not reveal pages after logout
    return resp


@app.context_processor
def inject_globals():
    return {"csrf_token": csrf_token, "max_mb": config.MAX_UPLOAD_MB}


def cleanup_uploads(max_age_hours=24):
    cutoff = time.time() - max_age_hours * 3600
    for p in config.UPLOAD_DIR.glob("*"):
        if p.is_file() and p.name != ".gitkeep" and p.stat().st_mtime < cutoff:
            try:
                p.unlink()
            except OSError:
                pass


def is_jpeg_or_png(data: bytes) -> bool:
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"


def load_result(uid):
    """Only the user who made the prediction may open it."""
    if not UID_RE.match(uid):
        abort(404)
    p = config.UPLOAD_DIR / f"{uid}.json"
    if not p.exists():
        abort(404)
    r = json.loads(p.read_text())
    if r.get("user_id") != session.get("user_id"):
        abort(404)
    return r


# ---------------------------------------------------------------- auth pages
@app.route("/")
def index():
    return redirect(url_for("upload" if "user_id" in session else "login"))


def _auth_page(mode):
    if "user_id" in session:
        return redirect(url_for("upload"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if mode == "register":
            if password != request.form.get("confirm", ""):
                flash("Passwords do not match.", "danger")
            else:
                ok, msg = auth.create_user(username, password)
                if ok:
                    flash("Account created. Please login.", "success")
                    return redirect(url_for("login"))
                flash(msg, "danger")
        else:
            if auth.locked(username):
                flash("Too many attempts. Try again in a minute.", "danger")
            else:
                user = auth.verify_user(username, password)
                if user:
                    session.clear()
                    session["user_id"], session["username"] = user["id"], user["username"]
                    return redirect(url_for("upload"))
                flash("Wrong username or password.", "danger")
    return render_template("login.html", mode=mode)


@app.route("/login", methods=["GET", "POST"])
def login():
    return _auth_page("login")


@app.route("/register", methods=["GET", "POST"])
def register():
    return _auth_page("register")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- the 5 steps
@app.route("/upload")                      # step 2
@login_required
def upload():
    return render_template("upload.html", model_ready=model_ready())


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Please choose an image first.", "danger")
        return redirect(url_for("upload"))
    if Path(file.filename).suffix.lower() not in config.VALID_EXTENSIONS:
        flash("Only JPG, JPEG or PNG images are allowed.", "danger")
        return redirect(url_for("upload"))
    data = file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        flash(f"File is too large (max {config.MAX_UPLOAD_MB} MB).", "danger")
        return redirect(url_for("upload"))
    if not is_jpeg_or_png(data):
        flash("This file is not a valid JPEG or PNG image.", "danger")
        return redirect(url_for("upload"))
    try:
        img = preprocessing.read_image_bytes(data)
    except ValueError:
        flash("This image could not be read. It may be corrupted.", "danger")
        return redirect(url_for("upload"))
    if min(img.shape[:2]) < config.MIN_IMAGE_SIDE:
        flash(f"Image is too small (min {config.MIN_IMAGE_SIDE} px per side).", "danger")
        return redirect(url_for("upload"))
    try:
        model = get_model()
    except ModelNotTrainedError as e:
        flash(str(e), "warning")
        return redirect(url_for("upload"))

    cleanup_uploads()
    uid = uuid.uuid4().hex
    try:
        result = model.predict(img, uid)
    except Exception:  # noqa: BLE001
        app.logger.exception("Prediction failed")
        flash("Could not analyse this image. Please try another photo.", "danger")
        return redirect(url_for("upload"))
    result["user_id"] = session["user_id"]
    (config.UPLOAD_DIR / f"{uid}.json").write_text(json.dumps(result))
    return redirect(url_for("processing", uid=uid))


@app.route("/processing/<uid>")            # step 3
@login_required
def processing(uid):
    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else None
    return render_template("processing.html", r=load_result(uid), m=metrics)


@app.route("/result/<uid>")                # step 4
@login_required
def result(uid):
    return render_template("result.html", r=load_result(uid))


@app.route("/performance")                 # step 5
@login_required
def performance():
    metrics = None
    if config.METRICS_PATH.exists():
        metrics = json.loads(config.METRICS_PATH.read_text())
    return render_template("performance.html", m=metrics)


@app.route("/media/<name>")
@login_required
def media(name):
    m = MEDIA_RE.match(name)
    if not m:
        abort(404)
    load_result(m.group(1))                # ownership check
    return send_from_directory(config.UPLOAD_DIR, name)


# ---------------------------------------------------------------- errors
@app.errorhandler(400)
def bad_request(_):
    return render_template("error.html", code=400, message="Session expired. Please go back and try again."), 400


@app.errorhandler(413)
def too_large(_):
    flash(f"File is too large (max {config.MAX_UPLOAD_MB} MB).", "danger")
    return redirect(url_for("upload" if "user_id" in session else "login"))


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, message="Page not found."), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
