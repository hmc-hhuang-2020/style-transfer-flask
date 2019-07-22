from tensorflow.python.keras import models
# import cv2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.preprocessing import image as kp_image
import tensorflow.contrib.eager as tfe
import tensorflow as tf
import functools
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize(
        (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

    img = kp_image.img_to_array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img


def imshow(img, title=None):
    # Remove the batch dimension
    out = np.squeeze(img, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def load_and_process_img_crop(path_to_img):
    img = load_img(path_to_img)
    img_copy = img.copy()
    black_pixels_mask = np.any(np.logical_or(
        img == [0, 0, 0], img == [255, 255, 255]), axis=-1)
    non_black_pixels_mask = np.any(np.logical_and(
        img != [0, 0, 0], img != [255, 255, 255]),  axis=-1)
    img[black_pixels_mask] = [0, 0, 0]

    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # We make the image channels first
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    # height, width, num filters of each layer
    # We scale the loss at a given layer by the size of the feature map and the number of filters
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    # / (4. * (channels ** 2) * (width * height) ** 2)
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path):
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    # Load our images in
    content_image = load_and_process_img_crop(content_path)
    style_image = load_and_process_img(style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0]
                      for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0]
                        for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    style_weight, content_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    # Accumulate style losses from all layers
    # Here, we equally weight each contribution of each loss layer
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * \
            get_style_loss(comb_style[0], target_style)

    # Accumulate content losses from all layers
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * \
            get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    # We don't need to (or want to) train any layers of our model, so we set their
    # trainable to false.
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(
        model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature)
                           for style_feature in style_features]

    # Set initial image
    init_image = load_and_process_img_crop(content_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.compat.v1.train.AdamOptimizer(
        learning_rate=5, beta1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        print('Iteration: {}'.format(i))
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        # if i % display_interval == 0:
            print('Iteration: {}'.format(i))
        #     start_time = time.time()

        #     # Use the .numpy() method to get the concrete numpy array
        #     plot_img = init_image.numpy()
        #     plot_img = deprocess_img(plot_img)
        #     imgs.append(plot_img)
    #         IPython.display.clear_output(wait=True)
    #         IPython.display.display_png(Image.fromarray(plot_img))
    #         print('Iteration: {}'.format(i))
    #         print('Total loss: {:.4e}, '
    #               'style loss: {:.4e}, '
    #               'content loss: {:.4e}, '
    #               'tixme: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    # print('Total time: {:.4f}s'.format(time.time() - global_start))
    # IPython.display.clear_output(wait=True)
    # plt.figure(figsize=(14, 4))
    # for i, img in enumerate(imgs):
    #     plt.subplot(num_rows, num_cols, i+1)
    #     plt.imshow(img)
    #     plt.xticks([])
    #     plt.yticks([])

    return best_img, best_loss


def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final:
        plt.figure(figsize=(10, 10))

        fig = plt.imshow(best_img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig('applied.jpg', bbox_inches='tight',
                    pad_inches=0)
        plt.title('Output Image')
        plt.show()


def blending(content_path, original_path, style_path):
    content = load_img(content_path).astype('uint8')
    original = load_img(original_path).astype('uint8')

    original_copy = original  # <----- original image (or original path)
    styled = load_img(style_path).astype('uint8')  # <---- stylized image
    mask = styled
    styled = styled.astype(float)
    styled = styled.reshape(styled.shape[1], styled.shape[2], 3)
    original_copy = original_copy.reshape(
        original_copy.shape[1], original_copy.shape[2], 3)
    original_copy = original_copy.astype(float)
    mask[non_black_pixels_mask] = [255, 255, 255]
    m = mask
    m = m.reshape(content.shape[1], content.shape[2], 3)

    cv2.imwrite('bin-mask-str.jpg', m)
    blurSigma = 5
    m = cv2.imread('bin-mask-str.jpg').astype(float)/255.0
    m = cv2.GaussianBlur(m, (2*blurSigma+1, 2*blurSigma+1), blurSigma)

    # apply alpha blending
    style_layer = cv2.multiply(m, styled)
    regular_layer = cv2.multiply(1.0-m, original_copy)
    out = style_layer + regular_layer

    out = out.astype('uint8')
    output_str = 'final.jpg'
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_str, out)


# Set up some global values here
# content_path = os.path.abspath("../content/crop (0).jpg")
# original_path = os.path.abspath("../content/original (1).jpg")
# style_path = '/tmp/nst/Vassily_Kandinsky,_1913_-_Composition_7.jpg'

# best, best_loss = run_style_transfer(
#     content_path, style_path, num_iterations=500)
# Image.fromarray(best)
# im = Image.fromarray(best)
# im.save('styled.jpg')

# result = blending(content_path, original_path, style_path)
