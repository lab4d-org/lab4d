from flask import Flask, send_from_directory, url_for
import dominate
from dominate.tags import *
import glob
import os, sys
import pdb

sys.path.insert(0, os.getcwd())
from preprocess.libs.io import run_bash_command


app = Flask(__name__)


class model_viewer(dominate.tags.html_tag):
    pass


logdirs = sorted(glob.glob("logdir-12-05/*compose/"))
port = 8090
all_logdirs = []
for logdir in logdirs:
    if "compose" in logdir:
        run_bash_command(
            f"python lab4d/mesh_viewer.py --testdir {logdir}/export_0001/ --port {port}",
            background=True,
        )
        all_logdirs.append((logdir, port))
        port += 1


@app.route("/videos/<path:filename>")
def custom_video(filename):
    return send_from_directory(os.getcwd(), filename)


@app.route("/")
def home():
    # Your existing code to generate the HTML
    doc = dominate.document(title="Results visualizer of vid2sim")

    with doc.head:
        link(rel="stylesheet", href="style.css")
        script(type="text/javascript", src="script.js")

    with doc:
        script(
            type="module",
            src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js",
        )
        with div():
            attr(cls="header")
            h1("Results visualizer of vid2sim")

            # write a clickable link
            for logdir, port in all_logdirs:
                logname = logdir.strip("/").split("/")[-1]
                a("logname name: " + logname + " ")
                br()
                with video(id="%s1" % logname, height="240", controls=True):
                    video_url = url_for(
                        # "custom_video", filename="%s/export_0001/ref_rgb.mp4" % logdir
                        "custom_video",
                        filename="database/processed/Annotations/Full-Resolution/%s-0000/vis.mp4"
                        % logname[5:-8],
                    )
                    source(src=video_url, type="video/mp4")
                with video(id="%s2" % logname, height="240", controls=True):
                    video_url = url_for(
                        "custom_video",
                        filename="%s/export_0001/render-bone-compose-ref.mp4" % logdir,
                    )
                    source(src=video_url, type="video/mp4")
                br()
                with strong():
                    a("[Mesh Visualizer]", href="http://128.2.218.31:" + str(port))
                button("Play Both Videos", id="playButton%s" % logname)
                br()
                br()
                br()

                # play both videos
                script_text = """
                document.getElementById('playButton{0}').addEventListener('click', function() {{
                    var video1 = document.getElementById('{0}1');
                    var video2 = document.getElementById('{0}2');
                    console.log(video1.paused);  // Corrected print statement to console.log
                    if (video1.paused) {{
                        video1.play();
                        video2.play();
                    }} else {{
                        video1.pause();
                        video2.pause();
                    }}
                }});
                """.format(
                    logname
                )

                script(script_text, type="text/javascript")

    # Convert the DOM object to a string and return it
    html_str = doc.render()
    html_str = html_str.replace("model_viewer", "model-viewer")
    html_str = html_str.replace("camera_controls", "camera-controls")
    return html_str


if __name__ == "__main__":
    app.run(host="0.0.0.0")
