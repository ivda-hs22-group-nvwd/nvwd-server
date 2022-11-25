import sqlite3
import json
from flask import Flask, render_template, request

# Start with 'flask --debug run'

### INIT
app = Flask(__name__)
def get_db_connection():
    conn = sqlite3.connect('okcupid.sqlite')
    conn.row_factory = sqlite3.Row
    return conn



### ROUTES
@app.route('/template-example')
def template():
    conn = get_db_connection()
    persons = conn.execute('SELECT * FROM okcupid').fetchall()
    conn.close()
    return render_template('index.html', persons=persons)


## CLEAN
@app.route('/api/clean/index')
def clean_index():
    with sqlite3.connect('okcupid.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_clean')
        data = cursor.fetchall()
        return json.dumps(data)

@app.route('/api/clean/<int:id>')
def clean_row(id):
    with sqlite3.connect('okcupid.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_clean WHERE rowid = ?', [id])
        data = cursor.fetchall()
        return json.dumps(data)

@app.route('/api/clean/age/between/',methods = ['POST'])
def alean_age_between():
    if request.method == 'POST':
        age_from = request.form['age_from']
        age_to = request.form['age_to']
    
        with sqlite3.connect('okcupid.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM okcupid_clean WHERE age BETWEEN ? and ?', [age_from, age_to])
            data = cursor.fetchall()
            return json.dumps(data)

## STANDARDIZED
@app.route('/api/std/index')
def std_index():
    with sqlite3.connect('okcupid.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_std')
        data = cursor.fetchall()
        return json.dumps(data)

@app.route('/api/std/<int:id>')
def std_row(id):
    with sqlite3.connect('okcupid.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_std WHERE rowid = ?', [id])
        data = cursor.fetchall()
        return json.dumps(data)

## BOTH
@app.route('/api/<int:id>')
def row(id):
    with sqlite3.connect('okcupid.sqlite') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_clean WHERE rowid = ?', [id])
        clean = cursor.fetchall()
        
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM okcupid_std WHERE rowid = ?', [id])
        std = cursor.fetchall()
        
        return json.dumps(clean+std)
