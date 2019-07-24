import logging
import os
import sys
import time
from flask import (Flask, flash, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from google.cloud import storage
# from style_transfer import run_style_transfer
from PIL import Image
# from object_detection import load_object, show_selection, InferenceConfig
import mrcnn.model as modellib

from object_detection import load_object, show_selection_outlines,show_selection_crop, show_selection_inverse, InferenceConfig

# import tensorflow as tf
# tf.enable_eager_execution()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

CLOUD_STORAGE_BUCKET = 'style-input-images-1'

RESULTS = None

SHOW_OBJECTS = None

STYLE_URL = None

DETECT_URL = None

FINAL_URL = None

content_copy = None

CONTENT_URL = None


def image_to_array(image):
    # Restricts to RGB
    im_array = np.array(image)[:, :, :3]

    if np.max(im_array) <= 1.0:
        im_array = np.floor(im_array * 255).astype(np.uint8)

    return im_array


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


@app.route('/upload', methods=['POST'])
def upload():
    transfer_option = request.form.get('transfer_select')
    if transfer_option == 'whole':
        style_path = request.files['style_file']
        # url = upload_to_gcloud(style_path)
        content_path = request.files['image_file']
        best, best_loss = run_style_transfer(
            content_path, style_path, num_iterations=1)
        im = Image.fromarray(best)
        im.save('static/out/styled.jpg')
        styled_file = os.path.join(
            os.path.abspath(''), 'static/out/styled.jpg')
        url = upload_to_gcloud_name(styled_file, 'styled_image.jpg')
        return render_template('upload.html', image_url=url)
    elif transfer_option == 'object':
        # # Upload style image first
        # style_path = request.files['style_file']
        # STYLE_URL = upload_to_gcloud(style_path)
        # url = STYLE_URL
        # storage_client = storage.Client(project='amli-245518')
        # bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
        # destination_blob_name = 'style.jpg'
        # blob = bucket.blob(destination_blob_name)
        # blob.upload_from_file(style_path)
        content_path = request.files['image_file']
        content_path_copy = content_path
        print(content_path_copy)

        ROOT_DIR = os.path.abspath("../")
        MaskRCNN_DIR = os.path.abspath("../Mask_RCNN")

        global CONTENT_URL
        CONTENT_URL = upload_to_gcloud(content_path_copy)

        # sys.path.append(os.path.join(MaskRCNN_DIR, "samples/coco/"))

        # sys.path.append(MaskRCNN_DIR)  # To find local version of the library
        MODEL_DIR = os.path.join(MaskRCNN_DIR, "samples/coco/")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

        config = InferenceConfig()

        detection_model = modellib.MaskRCNN(
            mode="inference", model_dir=MODEL_DIR, config=config)
        detection_model.load_weights(COCO_MODEL_PATH, by_name=True)

        global RESULTS
        global SHOW_OBJECTS
        content = download_from_gcloud('maskrcnn_test.jpg')
        RESULTS, SHOW_OBJECTS = load_object(content, detection_model)
        # load_object(image_file, detection_model)

        # config = InferenceConfig()

        # model = modellib.MaskRCNN(
        #     mode="inference", model_dir=MODEL_DIR, config=config)
        # model.load_weights(COCO_MODEL_PATH, by_name=True)

        # results, show_objects = load_object(image_file)
        # RESULTS = results
        # SHOW_OBJECTS = show_objects

        # contour_outlines = show_selection(raw_input, filename, show_objects)
        url = upload_to_gcloud_name(SHOW_OBJECTS, 'all_objects.jpg')
        return render_template('object.html', image_url=url)


@app.route("/select", methods=['POST'])
def select():
    selection = request.form.get('chosen_objects')
    selection = [int(x) for x in " ".join(selection.split(",")).split()]

    # content = download_from_gcloud("maskrcnn_test.jpg")
    # content_path = '/./mnt/c/Aaron/Documents/Career/GoogleAMLI/Data/Image'
    content_path = download_from_gcloud('maskrcnn_test.jpg')

    print(SHOW_OBJECTS)
    print(RESULTS)
    # image = download_from_gcloud('all_objects.jpg')

    contour_outlines = show_selection_outlines(selection, content_path, RESULTS)
    # contour_outlines = show_selection(selection, image, RESULTS)
    location, background_image = show_selection_crop(
        selection, content_path, RESULTS)
    DETECT_URL = upload_to_gcloud_name(
        location, 'selected_objects.jpg')
    return render_template('crop.html', image_url=DETECT_URL)


@app.route("/transform", methods=['POST'])
def transform():
    # STYLE_URL = "/style-input-images-1/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
    # gcs_file = storage.open(STYLE_URL)
    # style_path = gcs_file.read()
    # gcs_file.close()
    # DETECT_URL = "/style-input-images-1/styled_image.jpg"
    # gcs_file = storage.open(DETECT_URL)
    # content_path = gcs_file.read()
    # gcs_file.close()
    style_path = download_from_gcloud("style.jpg")
    content_path = download_from_gcloud("selected_objects.jpg")
    # best, best_loss = run_style_transfer(
    #     DETECT_URL, STYLE_URL, num_iterations=1)

    from style_transfer import run_style_transfer
    best, best_loss = run_style_transfer(
        content_path, style_path, num_iterations=1)
    im = Image.fromarray(best)
    im.save('static/out/styled_final.jpg')
    styled_file = os.path.join(
        os.path.abspath(''), 'static/out/styled_final.jpg')

    url = upload_to_gcloud_name(styled_file, 'styled_final.jpg')
    return render_template('final.html', image_url=url)


@app.route("/download", methods=['GET'])
def download():
    final_url = download_from_gcloud("styled_final.jpg")
    # final = 'styled_final.jpg'
    # client = storage.Client()
    # bucket = client.get_bucket(CLOUD_STORAGE_BUCKET)
    # blob = bucket.blob(final)
    # filename = 'final.jpg'
    # blob.download_to_filename(filename)
    return render_template('home.html')


# @app.route('/upload', methods=['POST'])
# def upload():
#     # style_path = request.files['style_file']
#     # content_path = request.files['image_file']
#     # best, best_loss = run_style_transfer(
#     #     content_path, style_path, num_iterations=500)
#     # Image.fromarray(best)
#     # im = Image.fromarray(best)
#     # styled = im.save('styled.jpg')
#     image_file = request.files['image_file']

#     print(image_file)

#     # model = os.path.join(os.path.abspath(''), 'rcnn_model.pkl')
#     # show_objects = load_object(image_file, model)
#     # contour_outlines = show_selection(raw_input, filename, show_objects)

#     # To find local version
#     ROOT_DIR = os.path.abspath("../")
#     MaskRCNN_DIR = os.path.abspath("../Mask_RCNN")
#     # sys.path.append(os.path.join(MaskRCNN_DIR, "samples/coco/"))

#     # sys.path.append(MaskRCNN_DIR)  # To find local version of the library
#     MODEL_DIR = os.path.join(MaskRCNN_DIR, "samples/coco/")
#     COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

#     config = InferenceConfig()

#     detection_model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
#     detection_model.load_weights(COCO_MODEL_PATH, by_name=True)


#     load_object(image_file, detection_model)

#     # storage_client = storage.Client(project='amli-245518')
#     # bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
#     # destination_blob_name = 'test.jpg'
#     # blob = bucket.blob(destination_blob_name)
#     # blob.upload_from_filename(style_file.filename)
#     # print("Send file")

#     return render_template('upload.html')


# @app.route('/uploaded', methods=['GET'])
# def uploaded():
#     return render_template('upload.html')

#     style_file = request.files['style_file']
#     blob.upload_from_filename(style_file)

#     style_file = request.files['style_file']
#     image_file = request.files['image_file']
#     if(style_file and allowed_file(style_file.filename) and image_file and allowed_file(image_file.filename)):

#         # Get files and styled image
#         output_img = 'static/out/' + time.ctime().replace(' ', '_')+'.jpg'
#         style_image = stylize(style_file, image_file,
#                               output_img, "rcnn_model.pkl")

#         # S3 Bucket
#         bucketName = "style-transfer-web"

#         # S3 upload image
#         s3 = boto3.client('s3')
#         s3.put_object(Body=style_image, Bucket=bucketName,
#                       Key=output_img, ContentType='image/jpeg')

#         session['file'] = output_img
#         return(render_template("home.html"))

# @app.route('/upload/', methods=['POST', 'GET'])
# def upload():
#     if 'file' in session:
#         # S3 Bucket
#         bucketName = "style-transfer-web"
#         s3 = boto3.resource('s3')
#         obj = s3.Object(bucketName, session['file'])
#         obj.delete()
#         session.clear()
#     style_file = request.files['style_file']
#     image_file = request.files['image_file']
#     if(style_file and allowed_file(style_file.filename) and image_file and allowed_file(image_file.filename)):

#         # Get files and styled image
#         output_img = 'static/out/' + time.ctime().replace(' ', '_')+'.jpg'
#         style_image = stylize(style_file, image_file,
#                               output_img, "rcnn_model.pkl")

#         # S3 Bucket
#         bucketName = "style-transfer-web"

#         # S3 upload image
#         s3 = boto3.client('s3')
#         s3.put_object(Body=style_image, Bucket=bucketName,
#                       Key=output_img, ContentType='image/jpeg')

#         session['file'] = output_img
#         return(render_template("home.html"))


# @app.route('/submitted', methods=['POST'])
# def submitted_form():
#     """Process the uploaded file and upload it to Google Cloud Storage."""
#     # uploaded_file = request.files['file']

#     # if not uploaded_file:
#     #     return 'No file uploaded.', 400

#     # Create a Cloud Storage client.
#     gcs = storage.Client()

#     if request.method == 'POST':
#         file = request.files['file']
#         bucket_name = "style-input-images-1"
#         path = '/' + bucket_name + '/' + str(secure_filename(file.filename))
#         if file and allowed_file(file.filename):
#             try:
#                 with gcs.open(path, 'w', **options) as f:
#                     f.write(file.stream.read())
#                     print(jsonify({"success": True}))
#                 return jsonify({"success": True})
#             except Exception as e:
#                 logging.exception(e)
#                 return jsonify({"success": False})
#     return render_template('home.html')


# @app.route('/submitted', methods=['POST'])
# def upload():
#     if request.method == "POST":
#         file = request.files.get("file")
#         my_upload = storage.upload(file)

#             # some useful properties
#         name = my_upload.name
#         size = my_upload.size
#         url = my_upload.url
#         return render_template('home.html')


# @app.route('/submitted', methods=['POST'])
# def submitted_form():
#     # image = request.files['file']
#     # [END submitted]
#     # [START render_template]
#     if request.method == 'POST':
#         file = request.files['file']
#         extension = secure_filename(file.filename).rsplit('.', 1)[1]
#         options = {}
#         options['retry_params'] = gcs.RetryParams(backoff_factor=1.1)
#         options['content_type'] = 'image/' + extension
#         bucket_name = "style-input-images-1"
#         path = '/' + bucket_name + '/' + str(secure_filename(file.filename))
#         if file and allowed_file(file.filename):
#             try:
#                 with gcs.open(path, 'w', **options) as f:
#                     f.write(file.stream.read())
#                     print(jsonify({"success": True}))
#                 return jsonify({"success": True})
#             except Exception as e:
#                 logging.exception(e)
#                 return jsonify({"success": False})
#     return render_template('home.html')

# @app.route('/submitted', methods=['POST'])
# def submitted_form():
    # """Process the uploaded file and upload it to Google Cloud Storage."""
    # uploaded_file = request.files.get('file')

    # if not uploaded_file:
    #     return 'No file uploaded.', 400

    # # Create a Cloud Storage client.
    # gcs = storage.Client()

    # # Get the bucket that the file will be uploaded to.
    # bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # # Create a new blob and upload the file's content.
    # blob = bucket.blob(uploaded_file.filename)

    # blob.upload_from_string(
    #     uploaded_file.read(),
    #     content_type=uploaded_file.content_type
    # )

    # # The public URL can be used to directly access the uploaded file via HTTP.
    # return blob.public_url

# @app.route('/submitted', methods=['POST'])
# def submitted_form():
#     # image = request.files['file']
#     # [END submitted]
#     # [START render_template]
#     if request.method == 'POST':
#         file = request.files['file']
#         extension = secure_filename(file.filename).rsplit('.', 1)[1]
#         options = {}
#         options['retry_params'] = gcs.RetryParams(backoff_factor=1.1)
#         options['content_type'] = 'image/' + extension
#         bucket_name = "style-input-images-1"
#         path = '/' + bucket_name + '/' + str(secure_filename(file.filename))
#         if file and allowed_file(file.filename):
#             try:
#                 with gcs.open(path, 'w', **options) as f:
#                     f.write(file.stream.read())
#                     print(jsonify({"success": True}))
#                 return jsonify({"success": True})
#             except Exception as e:
#                 logging.exception(e)
#                 return jsonify({"success": False})
#     return render_template('home.html')
    # [END render_template]

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500


if __name__ == '__main__':
    app.run()
