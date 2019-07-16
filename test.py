from google.cloud import storage
import os
from flask import request, send_file
import tempfile
from object_detection import load_object

style_file = os.path.join(os.path.abspath(''), 'stylize.jpg')
model = os.path.join(os.path.abspath(''), 'rcnn_model.pkl')
show_objects = load_object(style_file, model)
# style_file = request.files['stylize.jpg']
# style_file = os.path.abspath('../Flask-STWA/stylize.jpg')
# print(os.path.join(os.path.abspath(''), 'stylize.jpg'))

CLOUD_STORAGE_BUCKET = 'style-input-images-1'


def upload():
    style_file = os.path.join(os.path.abspath(''), 'stylize.jpg')
    CLOUD_STORAGE_BUCKET = 'style-input-images-1'

    storage_client = storage.Client(project='amli-245518')
    bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
    destination_blob_name = 'stylize.jpg'
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(style_file)


def download():
    CLOUD_STORAGE_BUCKET = 'style-input-images-1'
    style_file = 'stylize.jpg'
    # style_file = os.path.abspath('../Flask-STWA/stylize.jpg')
    client = storage.Client()
    bucket = client.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(style_file)
    filename = 'new.jpg'
    blob.download_to_filename(filename)
    # return send_file(temp.name, attachment_filename=style_file)
