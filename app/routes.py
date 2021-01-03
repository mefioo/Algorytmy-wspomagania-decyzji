from flask import render_template
from app import app


@app.route('/')
@app.route('/main')
def main():
    return render_template('main.html', title='Strona główna')

@app.route('/contact')
def contact():
    return render_template('Contact.html', title='Kontakt')

@app.route('/result')
def result():
    return render_template('result.html', title='Wynik')
