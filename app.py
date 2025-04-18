import os

from flask import Flask, flash, render_template, request, redirect, Response, url_for, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pred as predict


app = Flask(__name__)


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'jpg', 'jpeg', 'wav', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Error Handling 404
@app.errorhandler(404)
def error(code):
    return render_template("index.html", error=True, code=code), 404


# Error Handling 500
@app.errorhandler(500)
def error(code):
    return render_template("index.html", error=True, code=code), 500


# Error Handling 400
@app.errorhandler(400)
def error(code):
    return render_template("index.html", error=True, code=code), 400


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            s = ""
            ext = filename.split('.')[-1]
            if ext in ['mp4']:
                s = predict.Video().predict(filepath)
            elif ext in ['wav']:
                s = predict.Audio().predict(filepath)
            elif ext in ['jpg', 'jpeg', 'png']:
                s = predict.Image().predictor(filepath)
            return render_template("index.html", status=s)
        return render_template("index.html")
    return render_template("index.html")

if __name__ == "__main__":
    app.run()

    
