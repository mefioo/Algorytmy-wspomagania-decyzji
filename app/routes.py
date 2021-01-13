from flask import render_template, request, redirect, flash, url_for, make_response
import os
from os.path import join, dirname, realpath
from app import app


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html', title='Strona główna')
    elif request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return render_template('main.html', title='Strona główna')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template('main.html', title='Strona główna')
        if not allowed_file(file.filename):
            flash('Wrong file extension')
            return render_template('main.html', title='Strona główna')
        if file and allowed_file(file.filename):
            filename = 'zdjecie' #secure_filename(file.filename)
            file_path = join(dirname(realpath(__file__)), app.config['UPLOAD_FOLDER'], filename)
            print("FILEPATH IS: "+file_path)
            if(os.path.exists(file_path)):
                os.remove(file_path)
            file.save(file_path)
            return redirect(url_for('result'))

@app.route('/contact')
def contact():
    return render_template('contact.html', title='Kontakt')

@app.route('/result')
def result():
    app.convo
    data = []
    response = make_response(render_template('result.html', title='Wynik', data=data))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response