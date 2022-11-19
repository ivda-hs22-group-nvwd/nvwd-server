import sqlite3
import json
from flask import Flask, render_template

# Start with "flask run"

### INIT
app = Flask(__name__)
def get_db_connection():
    conn = sqlite3.connect('okcupid.db')
    conn.row_factory = sqlite3.Row
    return conn



### ROUTES
@app.route('/template-example')
def template():
    conn = get_db_connection()
    persons = conn.execute('SELECT * FROM okcupid').fetchall()
    conn.close()
    return render_template('index.html', persons=persons)


@app.route('/index')
def index():
    with sqlite3.connect('okcupid.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM okcupid")
        data = cursor.fetchall()
        return json.dumps(data)

# data is normalized!
# Todo: Insert non-normalized data
@app.route('/age/between/<float:start>/<float:end>')
def agebetween(start, end):
    with sqlite3.connect('okcupid.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM okcupid WHERE age > ? and age < ?", [start, end])
        data = cursor.fetchall()
        return json.dumps(data)
