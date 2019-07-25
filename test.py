# from google.cloud import storage
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
import numpy as np
from six.moves.urllib.request import urlopen
import tarfile

# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """Downloads a blob from the bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)

#     blob.download_to_filename(destination_file_name)

#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))

# def DownloadCheckpointFiles(checkpoint_dir=os.path.abspath("")):
#     """Download checkpoint files if necessary."""
#     full_checkpoint = "https://download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz"
#     # url_prefix = 'http://download.magenta.tensorflow.org/models/' 
#     # checkpoints = ['multistyle-pastiche-generator-monet.ckpt', 'multistyle-pastiche-generator-varied.ckpt']
#     # for checkpoint in checkpoints:
#     #     full_checkpoint = os.path.join(checkpoint_dir, checkpoint)
#     # checkpoint_dir = 'arbitrary_style_transfer'
#     if not os.path.exists(checkpoint_dir):
#         print('Downloading {}'.format(full_checkpoint))
#         filename = full_checkpoint.split("/")[-1]
#         with open(filename, "wb") as f:
#             r = requests.get(full_checkpoint)
#             f.write(r.content)

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

DownloadCheckpointFiles()    
# unzip_tar_gz()

# DownloadCheckpointFiles()

# detections =[[ 0.5210921   0.68086916  0.6457864   0.81711274 55          0.99862325]
#  [ 0.40780655  0.58433187  0.52803445  0.71685034 55          0.99838376]
#  [ 0.41141948  0.30821186  0.5301136   0.43577647 55         0.9982496 ]
#  [ 0.5196079   0.53558177  0.64213705  0.66894466 55          0.99801254]
#  [ 0.41874808  0.44184786  0.5332428   0.57064563 55          0.99788   ]
#  [ 0.51584554  0.37531173  0.65214545  0.5233197  55          0.9976165 ]
#  [ 0.3246436   0.3725551   0.42766646  0.48901382 55          0.9958526 ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
#  [ 0.          0.          0.          0.          0.          0.        ]
# ]

# zero_index = np.where(detections[:, 4] == 0)[0]
# print(zero_index)

# # print(os.path.basename(
# #     'gs: // style-input-images-1/all_objects.jpg'))

# # style = "static/style_images/Vassily_Kandinsky,_1913_-_Composition_7.jpg"
# # content = "static/input_images/styled.jpg"
# # test = "arbitrary_image_stylization_with_weights \
# #   --checkpoint=arbitrary_style_transfer/model.ckpt \
# #   --output_dir=outputs \
# #   --style_images_paths="+style+"\
# #   --content_images_paths="+content+"\
# #   --image_size=256 \
# #   --content_square_crop=False \
# #   --style_image_size=256 \
# #   --style_square_crop=False \
# #   --logtostderr"
# # path = os.system(test)
# # print(path)

# # from object_detection import blending
# crop_path = 'static/blending/crop.jpg'
# original_path = 'static/blending/original.jpg'
# style_path = 'static/blending/original_stylized_Vassily_Kandinsky,_1913_-_Composition_7_0.jpg'

# print(os.path.join(os.path.abspath(""), style_path))
# print(blending(crop_path, original_path, style_path))

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
