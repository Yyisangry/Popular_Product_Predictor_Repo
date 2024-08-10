from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import pandas as pd
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure the upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load your trained model (assuming it's saved in a pickle file)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    raw_data_html = None
    processed_data_html = None
    processed_filename = None
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty part without a filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the file using the trained model
            raw_data_html, processed_data_html, processed_filename = process_file(file_path)


    return render_template('combined.html', raw_data=raw_data_html, processed_data=processed_data_html, filename=processed_filename)

def process_file(file_path):
    # Load the raw data
    df = pd.read_csv(file_path)
    
    # Apply model to the raw data
    y_pred = model.predict(df)
    df_pred = pd.DataFrame(y_pred)
    
    # Construct the processed file path
    processed_filename = 'processed_' + os.path.basename(file_path)
    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    print(f"\033[34mProcessed file path: {processed_file_path}\033[0m")  # Blue text
    # Save the transformed data
    df_pred.to_csv(processed_file_path, index=False)
    
    # Convert the processed data to HTML, store the original data
    raw_data_html = df.to_html(classes='table table-striped', index=False)
    processed_data_html = df_pred.to_html(classes='table table-striped', index=False)
    return raw_data_html, processed_data_html, processed_filename

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    print(f"\033[34mFile path for download: {file_path}\033[0m")  # Blue text
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found!", 404
    #return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

