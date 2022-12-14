from flask import Flask, request, send_file, after_this_request, jsonify
import os
import uuid
from shutil import rmtree
from server.pano_utils import AdaintEngine

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(APP_ROOT, "var/tmp/")

os.makedirs(ML_DIR, exist_ok=True)

adaint_engine = AdaintEngine()


@app.route("/enhance/<mode>", methods=["POST"])
def enhance(mode):
    temp_name = str(uuid.uuid4())
    temp_dir = os.path.join(ML_DIR, temp_name)
    os.makedirs(temp_dir)
    filename = get_image(temp_dir)

    if filename is None:
        return "Wrong file format.", 400

    print("File was passed to ML model.")
    enhanced_filename = adaint_engine.enhance(filename, temp_dir, mode)
    print("File was processed. Returning...")

    @after_this_request
    def remove_file(response):
        try:
            rmtree(temp_dir)
        except Exception as error:
            app.logger.error("Error removing temp directory", error)
        return response

    return send_file(enhanced_filename)


def get_image(temp_dir):

    file = request.files['image']
    filename = file.filename
    print("File name: {}".format(filename))

    # file support verification
    ext = os.path.splitext(filename)[1].lower()
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return None

    # save file
    destination = os.path.join(temp_dir, filename)
    print("File saved to to:", destination)
    file.save(destination)

    return destination


if __name__ == "__main__":
    app.run(host="0.0.0.0")
