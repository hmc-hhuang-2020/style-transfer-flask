import logging
import os
import sys
import time
from flask import (Flask, flash, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from PIL import Image
import mrcnn.model as modellib
from mrcnn import utils

from object_detection import load_object, show_selection_outlines, show_selection_crop, show_selection_inverse, InferenceConfig, blending
from werkzeug.utils import secure_filename
from six.moves.urllib.request import urlopen
import tarfile


app = Flask(__name__)

RESULTS = None

SHOW_OBJECTS = None

STYLE_URL = None

CONTENT_URL = None

LOCATION = None
SELECTION = None

ROOT_DIR = os.path.abspath("")
MaskRCNN_DIR = ROOT_DIR

MODEL_DIR = os.path.join(MaskRCNN_DIR, "coco.py")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

config = InferenceConfig()
detection_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
detection_model.load_weights(COCO_MODEL_PATH, by_name=True)
detection_model.keras_model._make_predict_function()

def DownloadCheckpointFiles(checkpoint_dir=os.path.abspath("")):
    """Download checkpoint files if necessary """
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
    """Upzip checkpoint files """
    tf = tarfile.open('arbitrary_style_transfer.tar.gz',"r:gz")
    tf.extractall()
    tf.close()

def upload_style_content_images(style,content):
    """ Upload style image to style_images folder 
    and content image to input_images folder """
    style_name = secure_filename(style.filename)
    style_path = os.path.join('static/style_images', style_name)
    style.save(style_path)
    content_name = secure_filename(content.filename)
    content_path = os.path.join('static/input_images', content_name)
    content.save(content_path)
    return style_path, content_path

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    transfer_option = request.form.get('transfer_select')
    # Set global variable to access across different pages 
    global STYLE_URL, CONTENT_URL
    global RESULTS, SHOW_OBJECTS
    global LOCATION
    global SELECTION
    # Directly transform the whole image
    if transfer_option == 'whole':
        style = request.files['style_file']
        content = request.files['image_file']
        STYLE_URL, CONTENT_URL = upload_style_content_images(style,content)

        content_img_name = os.path.basename(CONTENT_URL)[:-4]
        style_img_name = os.path.basename(STYLE_URL)[:-4]

        # Run 100% style transfer with arbitrary_image_stylization model
        out = "arbitrary_image_stylization_with_weights \
        --checkpoint=arbitrary_style_transfer/model.ckpt \
        --output_dir=static/final \
        --style_images_paths="+STYLE_URL+"\
        --content_images_paths="+CONTENT_URL+"\
        --image_size=512 \
        --content_square_crop=False \
        --style_image_size=512 \
        --style_square_crop=False \
        --logtostderr"
        os.system(out)
        path = 'static/final/'+('%s_stylized_%s_0.jpg' %
                              (content_img_name, style_img_name))
        return render_template('upload.html', image_url=path)
    # Transform the whole image with different weights of transfer
    elif transfer_option == 'adjust':
        style = request.files['style_file']
        content = request.files['image_file']
        STYLE_URL, CONTENT_URL = upload_style_content_images(style,content)

        content_img_name = os.path.basename(CONTENT_URL)[:-4]
        style_img_name = os.path.basename(STYLE_URL)[:-4]

        # Run different weights of style transfer from 20% to 100%
        INTERPOLATION_WEIGHTS='[0.2,0.4,0.6,0.8,1.0]'
        output = "arbitrary_image_stylization_with_weights \
        --checkpoint=arbitrary_style_transfer/model.ckpt \
        --output_dir=static/final \
        --style_images_paths="+STYLE_URL+"\
        --content_images_paths="+CONTENT_URL+"\
        --image_size=512 \
        --content_square_crop=False \
        --style_image_size=512 \
        --style_square_crop=False \
        --interpolation_weights="+INTERPOLATION_WEIGHTS+"\
        --logtostderr"
        os.system(output)
        changed_paths = []
        for i in range(5):
            changed_paths.append('static/final/' + ('%s_stylized_%s_%d.jpg' %
                                        (content_img_name, style_img_name,i)))  
        return render_template('wholeOptions.html', image_url=changed_paths)
    # Object Detection
    elif transfer_option == 'object':
        SELECTION = 'object'
        style = request.files['style_file']
        content = request.files['image_file']
        STYLE_URL, CONTENT_URL = upload_style_content_images(style,content)

        # Run Object Detection
        RESULTS, SHOW_OBJECTS = load_object(CONTENT_URL, detection_model)
        return render_template('object.html', image_url=SHOW_OBJECTS)
    # Inverse Object Detection
    elif transfer_option == 'inverse':
        SELECTION = 'inverse'
        style = request.files['style_file']
        content = request.files['image_file']
        STYLE_URL, CONTENT_URL = upload_style_content_images(style,content)

        # Run Object Detection
        RESULTS, SHOW_OBJECTS = load_object(CONTENT_URL, detection_model)
        return render_template('object.html', image_url=SHOW_OBJECTS)


@app.route("/select", methods=['POST'])
def select():
    global LOCATION
    # Run different crop strategies according to selections
    selection = request.form.get('chosen_objects')
    selection = [int(x) for x in " ".join(selection.split(",")).split()]
    contour_outlines = show_selection_outlines(
        selection, CONTENT_URL, RESULTS)
    if SELECTION == 'object':
        location, background_image = show_selection_crop(
            selection, CONTENT_URL, RESULTS)
        LOCATION = location
    elif SELECTION == 'inverse':
        location, background_image = show_selection_inverse(
            selection, CONTENT_URL, RESULTS)
        LOCATION = location
    return render_template('crop.html', image_url=contour_outlines)

@app.route("/transform", methods=['POST'])
def transform():
    # Transform object detection with options to adjust weights
    scale_option = request.form.get('scale')
    content_img_name = os.path.basename(LOCATION)[:-4]
    style_img_name = os.path.basename(STYLE_URL)[:-4]
    # Direct Transformation with 100% style transfer
    if scale_option == 'no':
        output = "arbitrary_image_stylization_with_weights \
            --checkpoint=arbitrary_style_transfer/model.ckpt \
            --output_dir=static/final \
            --style_images_paths="+STYLE_URL+"\
            --content_images_paths="+LOCATION+"\
            --image_size=512 \
            --content_square_crop=False \
            --style_image_size=512 \
            --style_square_crop=False \
            --logtostderr"
        os.system(output)
        changed_path = 'static/final/' + ('%s_stylized_%s_0.jpg' %
                                        (content_img_name, style_img_name))
        output_str = blending(LOCATION, CONTENT_URL, changed_path)
        return render_template('final.html', image_url=output_str)
    # Transformation adjustable from 20% to 100% weights
    elif scale_option == 'yes':
        INTERPOLATION_WEIGHTS='[0.2,0.4,0.6,0.8,1.0]'
        outputs = "arbitrary_image_stylization_with_weights \
            --checkpoint=arbitrary_style_transfer/model.ckpt \
            --output_dir=static/final \
            --style_images_paths="+STYLE_URL+"\
            --content_images_paths="+LOCATION+"\
            --image_size=512 \
            --content_square_crop=False \
            --style_image_size=512 \
            --style_square_crop=False \
            --interpolation_weights="+INTERPOLATION_WEIGHTS+"\
            --logtostderr"
        os.system(outputs)
        changed_paths = []
        for i in range(5):
            changed_paths.append('static/final/' + ('%s_stylized_%s_%d.jpg' %
                                        (content_img_name, style_img_name,i)))                                                         
        return render_template('options.html',image_url=changed_paths)

@app.route("/blend", methods=['POST'])
def blend():
    # Blend the transformed cropped image with original image
    content_img_name = os.path.basename(LOCATION)[:-4]
    style_img_name = os.path.basename(STYLE_URL)[:-4]
    select_number = request.form.get('weightScale')
    changed_path_select = 'static/final/' + ('%s_stylized_%s_%s.jpg' %
                                        (content_img_name, style_img_name,select_number))
    output_str_select = blending(LOCATION, CONTENT_URL, changed_path_select)
    return render_template('final.html',image_url=output_str_select)


@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


if __name__ == '__main__':
    DownloadCheckpointFiles()    
    app.run(threaded=True)
