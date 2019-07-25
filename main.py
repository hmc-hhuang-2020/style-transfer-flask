import logging
import os
import sys
import time
from flask import (Flask, flash, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from google.cloud import storage
from PIL import Image
import mrcnn.model as modellib
from mrcnn import utils

from object_detection import load_object, show_selection_outlines, show_selection_crop, show_selection_inverse, InferenceConfig, blending
from werkzeug.utils import secure_filename
from six.moves.urllib.request import urlopen
import tarfile

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

CLOUD_STORAGE_BUCKET = 'style-input-images-1'

RESULTS = None

SHOW_OBJECTS = None

STYLE_URL = None

DETECT_URL = None

FINAL_URL = None

CONTENT_URL = None

LOCATION = None

# ROOT_DIR = os.path.abspath("")
# MaskRCNN_DIR = os.path.abspath("Mask_RCNN")
# MODEL_DIR = os.path.join(MaskRCNN_DIR, "samples/coco/")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# config = InferenceConfig()

# detection_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# detection_model.load_weights(COCO_MODEL_PATH, by_name=True)

def DownloadCheckpointFiles(checkpoint_dir=os.path.abspath("")):
    """Download checkpoint files if necessary."""
    url_prefix = 'http://download.magenta.tensorflow.org/models/' 
    checkpoints = ['arbitrary_style_transfer.tar.gz']
    path = 'arbitrary_style_transfer'
    for checkpoint in checkpoints:
        full_checkpoint = os.path.join(checkpoint_dir, checkpoint)
        if not os.path.exists(path):
            print('Downloading {}'.format(full_checkpoint))
            response = urlopen(url_prefix + checkpoint)
            data = response.read()
            with open(full_checkpoint, 'wb') as fh:
                fh.write(data)
            unzip_tar_gz()

def unzip_tar_gz():
    tf = tarfile.open('arbitrary_style_transfer.tar.gz',"r:gz")
    tf.extractall()
    tf.close()


def upload_to_gcloud(file):
    # style_path = request.files['style_file']
    storage_client = storage.Client(project='amli-245518')
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(file.filename)
    blob.upload_from_file(file)
    return blob.public_url


def upload_to_gcloud_name(file, destination_blob_name):
    # style_path = request.files['style_file']
    storage_client = storage.Client(project='amli-245518')
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file)
    return blob.public_url


def download_from_gcloud(filename):
    # style_file = os.path.abspath('../Flask-STWA/stylize.jpg')
    client = storage.Client()
    bucket = client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(filename)
    temp = 'static/out/'+filename
    blob.download_to_filename(temp)
    # return send_file(temp.name, attachment_filename=style_file)
    return temp


def allowed_file(filename):
    return('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/test')
def test():
    return render_template('object.html')


@app.route('/upload', methods=['POST'])
def upload():
    transfer_option = request.form.get('transfer_select')
    if transfer_option == 'whole':
        style = request.files['style_file']
        style_name = secure_filename(style.filename)
        style_path = os.path.join('static/style_images', style_name)

        style.save(style_path)
        content = request.files['image_file']
        content_name = secure_filename(content.filename)
        content_path = os.path.join('static/input_images', content_name)
        content.save(content_path)

        content_img_name = os.path.basename(content_path)[:-4]
        style_img_name = os.path.basename(style_path)[:-4]
        test = "arbitrary_image_stylization_with_weights \
        --checkpoint=arbitrary_style_transfer/model.ckpt \
        --output_dir=static/out \
        --style_images_paths="+style_path+"\
        --content_images_paths="+content_path+"\
        --image_size=256 \
        --content_square_crop=False \
        --style_image_size=256 \
        --style_square_crop=False \
        --logtostderr"
        os.system(test)
        path = 'static/out/'+('%s_stylized_%s_0.jpg' %
                              (content_img_name, style_img_name))
        return render_template('upload.html', image_url=path)
    elif transfer_option == 'object':
        # # Upload style image first
        global STYLE_URL, CONTENT_URL
        style = request.files['style_file']
        style_name = secure_filename(style.filename)
        style_path = os.path.join('static/style_images', style_name)
        style.save(style_path)
        STYLE_URL = style_path
        content = request.files['image_file']
        content_name = secure_filename(content.filename)
        content_path = os.path.join('static/input_images', content_name)
        content.save(content_path)
        CONTENT_URL = content_path

        ROOT_DIR = os.path.abspath("")
        MaskRCNN_DIR = ROOT_DIR

        MODEL_DIR = os.path.join(MaskRCNN_DIR, "coco.py")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = InferenceConfig()

        detection_model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config)
        detection_model.load_weights(COCO_MODEL_PATH, by_name=True)

        global RESULTS
        global SHOW_OBJECTS
        RESULTS, SHOW_OBJECTS = load_object(CONTENT_URL, detection_model)
        return render_template('object.html', image_url=SHOW_OBJECTS)


@app.route("/select", methods=['POST'])
def select():
    selection = request.form.get('chosen_objects')
    selection = [int(x) for x in " ".join(selection.split(",")).split()]

    contour_outlines = show_selection_outlines(
        selection, CONTENT_URL, RESULTS)

    location, background_image = show_selection_crop(
        selection, CONTENT_URL, RESULTS)
    global LOCATION
    LOCATION = location
    return render_template('crop.html', image_url=contour_outlines)


@app.route("/transform", methods=['POST'])
def transform():
    content_img_name = os.path.basename(LOCATION)[:-4]
    style_img_name = os.path.basename(STYLE_URL)[:-4]

    test = "arbitrary_image_stylization_with_weights \
        --checkpoint=arbitrary_style_transfer/model.ckpt \
        --output_dir=static/final \
        --style_images_paths="+STYLE_URL+"\
        --content_images_paths="+LOCATION+"\
        --image_size=256 \
        --content_square_crop=False \
        --style_image_size=256 \
        --style_square_crop=False \
        --logtostderr"
    os.system(test)
    changed_path = 'static/final/' + ('%s_stylized_%s_0.jpg' %
                                      (content_img_name, style_img_name))
    output_str = blending(LOCATION, CONTENT_URL, changed_path)
    return render_template('final.html', image_url=output_str)


@app.route("/download", methods=['GET'])
def download():
    return render_template('home.html')

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


if __name__ == '__main__':
    DownloadCheckpointFiles()    
    app.run(threaded=True)
