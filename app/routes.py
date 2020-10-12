from flask import render_template
from app import app


@app.route('/')
@app.route('/main')
def main():
    return render_template('main.html', title='Strona główna')
