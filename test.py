from google.cloud import storage
import os
from flask import request, send_file
import tempfile
# import cloudstorage as gcs
# from google.appengine.api import images, app_identity

# STYLE_URL = "/style-input-images-1/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
# f = gcs.open(STYLE_URL)
# data = f.read()
# print(data)
# from object_detection import load_object
# from style_transfer import run_style_transfer

import os

# print(os.path.basename(
#     'gs: // style-input-images-1/all_objects.jpg'))

# style = "static/style_images/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
# content = "static/input_images/styled.jpg"
# test = "arbitrary_image_stylization_with_weights \
#   --checkpoint=arbitrary_style_transfer/model.ckpt \
#   --output_dir=outputs \
#   --style_images_paths="+style+"\
#   --content_images_paths="+content+"\
#   --image_size=256 \
#   --content_square_crop=False \
#   --style_image_size=256 \
#   --style_square_crop=False \
#   --logtostderr"
# path = os.system(test)
# print(path)

from object_detection import blending
crop_path = 'static/blending/crop1.jpg'
original_path = 'static/blending/original.jpg'
style_path = 'static/blending/original_stylized_Vassily_Kandinsky,_1913_-_Composition_7_0.jpg'
print(blending(crop_path, original_path, style_path))

# test1 = None


# def test():
#     global test1
#     test1 = 'TEST'
#     return


# if __name__ == "__main__":
#     test()
#     print(test1)


# style_file = os.path.join(os.path.abspath(''), 'stylize.jpg')
# model = os.path.join(os.path.abspath(''), 'rcnn_model.pkl')
# show_objects = load_object(style_file, model)
# style_file = request.files['styled.jpg']
# print(style_file)
# style_file = os.path.abspath('../Flask-STWA/stylize.jpg')
# print(os.path.join(os.path.abspath(''), 'stylize.jpg'))

# CLOUD_STORAGE_BUCKET = 'style-input-images-1'


# def upload_to_gcloud(file, destination_blob_name):
#     # style_path = request.files['style_file']
#     storage_client = storage.Client(project='amli-245518')
#     bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(file)
#     return blob.public_url


# style_file = os.path.join(os.path.abspath(''), 'styled.jpg')
# url = upload_to_gcloud(style_file, 'styled.jpg')
# print(url)

# style_file = os.path.join(os.path.abspath(''), 'stylize.jpg')
# CLOUD_STORAGE_BUCKET = 'style-input-images-1'
# best, best_loss = run_style_transfer(
#     content_path, style_path, num_iterations=3)
# storage_client = storage.Client(project='amli-245518')
# bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
# destination_blob_name = 'styled-test.jpg'
# blob = bucket.blob(destination_blob_name)
# blob.upload_from_filename(style_file)


# def upload():
#     style_file = os.path.join(os.path.abspath(''), 'stylize.jpg')
#     CLOUD_STORAGE_BUCKET = 'style-input-images-1'
#     best, best_loss = run_style_transfer(
#         content_path, style_path, num_iterations=3)
#     storage_client = storage.Client(project='amli-245518')
#     bucket = storage_client.get_bucket(CLOUD_STORAGE_BUCKET)
#     destination_blob_name = 'stylize.jpg'
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(style_file)


# def download():
#     CLOUD_STORAGE_BUCKET = 'style-input-images-1'
#     style_file = 'stylize.jpg'
#     # style_file = os.path.abspath('../Flask-STWA/stylize.jpg')
#     client = storage.Client()
#     bucket = client.get_bucket(CLOUD_STORAGE_BUCKET)
#     blob = bucket.blob(style_file)
#     filename = 'new.jpg'
#     blob.download_to_filename(filename)
#     # return send_file(temp.name, attachment_filename=style_file)
