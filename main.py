from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():    
    return render_template('home.html')

@app.route('/submitted', methods=['POST'])
def submitted_form():
    image = request.files['file']
    # [END submitted]
    # [START render_template]
    return render_template('home.html')
    # [END render_template]

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500

# @app.route('/')
# def hello():
#     return "Hello World!"

if __name__ == '__main__':
    app.run()