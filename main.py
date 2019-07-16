import logging
import os
import time
from flask import (Flask, flash, make_response, redirect, render_template,
                   request, send_file, session, url_for)
from google.cloud import storage
from style_transfer import run_style_transfer
from PIL import Image
from object_detection import load_object

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

CLOUD_STORAGE_BUCKET = 'style-input-images-1'


def allowed_file(filename):
    return('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


# @app.route("/upload")
# def upload():
#     uploadUri = blobstore.create_upload_url(
#         '/submit', gs_bucket_name=CLOUD_STORAGE_BUCKET)
#     return render_template('upload.html', uploadUri=uploadUri)


# @app.route("/submit", methods=['POST'])
# def submit():
#     if request.method == 'POST':
#         f = request.files['style_file']
#         header = f.headers['Content-Type']
#         parsed_header = parse_options_header(header)
#         blob_key = parsed_header[1]['blob-key']
#     return blob_key


# @app.route("/img/<bkey>")
# def img(bkey):
#     blob_info = blobstore.get(bkey)
#     response = make_response(blob_info.open().read())
#     response.headers['Content-Type'] = blob_info.content_type
#     return response


@app.route('/upload', methods=['POST'])
def upload():
    # style_path = request.files['style_file']
    # content_path = request.files['image_file']
    # best, best_loss = run_style_transfer(
    #     content_path, style_path, num_iterations=500)
    # Image.fromarray(best)
    # im = Image.fromarray(best)
    # styled = im.save('styled.jpg')
    image_file = request.files['image_file']
    model = os.path.join(os.path.abspath(''), 'rcnn_model.pkl')
    show_objects = load_object(image_file, model)
    # contour_outlines = show_selection(raw_input, filename, show_objects)
    storage_client = storage.Client(project='amli-245518')
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    destination_blob_name = 'test.jpg'
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(style_file.filename)
    print("Send file")

    return render_template('upload.html')


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
