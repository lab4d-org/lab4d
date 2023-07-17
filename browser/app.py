# WIP by Gengshan Yang
# python browser/app.py 'database/processed/Annotations/Full-Resolution/cat-85-*/vis.mp4'
# python browser/app.py 'logdir/dog-98-category-comp/renderings_00*/xyz.mp4'
# or python browser/app.py and type in string
from flask import Flask, render_template, request, send_from_directory
import os
import sys
import glob

app = Flask(__name__)


def get_files(path):
    matched_files = sorted(glob.glob(path))
    return matched_files


@app.route("/", methods=["GET", "POST"])
def index():
    files = []
    if request.method == "POST":
        path = request.form.get("path")

    elif len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = ""
    files = get_files(path)
    return render_template("index.html", files=files)


@app.route("/logdir/<path:filename>", methods=["GET"])
def get_logdir_file(filename):
    return send_from_directory(os.getcwd(), filename)


@app.route("/database/<path:filename>", methods=["GET"])
def get_database_file(filename):
    return send_from_directory(os.getcwd(), filename)


if __name__ == "__main__":
    app.run(debug=True)
