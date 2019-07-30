# Localized Style Transfer

Localized Style Transfer is an application that performs style transfer while allowing the user control over which portion of the image the style transfer gets applied to. The user provides two images: a **style image** to emulate and a **content image** to apply the style to. The application provides the user with a list of objects detected to create a mask and control where the style transfer gets applied to, leaving the rest of the image intact.

## Installation

In order to host and run the web application, follow these steps:
1. Clone the repository.
2. For Ubuntu, install `libasound2-dev`: `sudo apt-get install libasound2-dev`
3. Create a virtual environment to install dependencies: `python3 -m venv path/to/my_venv`
4. Activate the virtual enviroment created: `source activate` 
5. Install dependencies from `requirements.txt`: `pip install -r requirements.txt`
6. To install dependencies needed to access MS COCO:

   On Linux, run `pip install git+https://github.com/waleedka/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI` (Try pip3 if that didn't work)
   
   On Windows, run `pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`
   
   
7. Run `pip install magenta` to install magenta machine learning packages. 

## Usage

1. Run the web application using the command `python main.py`
2. The script will automatically download mask_rcnn_coco.h5 file for the object detection model and arbitrary_style_transfer folder for the style transfer model.
3. Open a web browser and enter the address `localhost:5000` or the address appeared on the running on `address` line.
4. Select a style image and a content image.
5. Choose whether to apply the style transfer to:  
   - the whole image 
   - the whole image with adjustable style weights
   - to selected objects
   - inverse of the selected objects.
6. In the case of the third and forth options, pick objects by inputing a comma or space separated list of numbers. Each number corresponds to the label number appeared at th beginning of the detected captions. Enter 1000 to select all detected objects. 
