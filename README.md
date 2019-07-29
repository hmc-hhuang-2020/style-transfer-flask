# Localized Style Transfer

Localized Style Transfer is an application that performs style transfer while allowing the user control over which portion of the image the style transfer gets applied to. The user provides two images: a **style image** to emulate and a **content image** to apply the style to. The application provides the user with a list of objects to create a mask and control where the style transfer gets applied to, leaving the rest of the image intact.

## Installation

In order to host and run the web application, follow these steps:
1. Clone the repository.
2. Install `libasound2-dev`: `sudo apt-get install libasound2-dev`
3. Create a virtual environment to install dependencies: `python3 -m venv path/to/my_venv`
4. Install dependencies from `requirements.txt`: `pip install -r requirements.txt`

## Usage

1. Run the web application using the command `python main.py`
2. Select a style image and content image.
3. Choose whether to apply the style transfer to:  
   - the whole image 
   - to selected objects
   - inverse of the selected objects.
4. In the case of the second and third options, pick objects by inputing a comma separated list of numbers.