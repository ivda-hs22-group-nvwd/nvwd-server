import sqlite3
import json
from flask import Flask, render_template, request

### QUICK TEST WILL BE REFACTORED
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

continuous_cols = ['age', 'height']
categorical_cols = ['body_type', 'drinks', 'drugs', 'income', 'job', 'orientation', 'sex', 'smokes',
'diet','diet_modifier',
'education_status', 'education_institution',
'offspring_status', 'offspring_future',
'pets_cats', 'pets_dogs',
'religion_type', 'religion_modifier',
'sign', 'sign_modifier']
ethnities_cols = ['ethnicities_middle_eastern', 'ethnicities_hispanic_/_latin',
       'ethnicities_white', 'ethnicities_indian', 'ethnicities_other',
       'ethnicities_asian', 'ethnicities_black', 'ethnicities_native_american',
       'ethnicities_pacific_islander']
speaks_cols = ['speaks_english', 'speaks_spanish', 'speaks_french', 'speaks_c++',
       'speaks_chinese', 'speaks_tagalog', 'speaks_portuguese',
       'speaks_japanese', 'speaks_russian', 'speaks_ukrainian',
       'speaks_sanskrit', 'speaks_thai', 'speaks_hindi', 'speaks_sign',
       'speaks_swedish', 'speaks_german', 'speaks_italian', 'speaks_arabic',
       'speaks_latin', 'speaks_other', 'speaks_hebrew', 'speaks_hawaiian',
       'speaks_korean', 'speaks_ancient', 'speaks_vietnamese',
       'speaks_indonesian', 'speaks_latvian', 'speaks_hungarian',
       'speaks_lisp', 'speaks_swahili', 'speaks_rotuman', 'speaks_czech',
       'speaks_yiddish', 'speaks_greek', 'speaks_catalan', 'speaks_croatian',
       'speaks_farsi', 'speaks_icelandic', 'speaks_tamil', 'speaks_serbian',
       'speaks_esperanto', 'speaks_norwegian', 'speaks_bengali',
       'speaks_dutch', 'speaks_urdu', 'speaks_irish', 'speaks_welsh',
       'speaks_sign_language', 'speaks_khmer', 'speaks_cebuano',
       'speaks_afrikaans', 'speaks_albanian', 'speaks_romanian',
       'speaks_polish', 'speaks_turkish', 'speaks_finnish']
mapper = DataFrameMapper(
  [([continuous_col], StandardScaler()) for continuous_col in continuous_cols] +
  [(categorical_col, LabelEncoder()) for categorical_col in categorical_cols] +
  [(ethnities_col, LabelEncoder()) for ethnities_col in ethnities_cols] +
  [(speaks_col, LabelEncoder()) for speaks_col in speaks_cols],
  df_out=True 
)

df_clean = pd.read_csv('../pipeline/data/cleaned.csv')
sample = df_clean.iloc[:1]
df_std = np.round(mapper.fit_transform(df_clean.copy()),2)


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


## DEV-LOGIC
@app.route('/api/dev',methods = ['POST'])
def user_input():
    if request.method == 'POST':
        
        
        # TODO: THis mocks fake input
        result_orig = sample.to_json(orient="split")
        parsed_orig = json.loads(result_orig) 
        
        sample_std = np.round(mapper.transform(sample), 2) # standardize
        result_std = sample_std.to_json(orient="split") 
        parsed_std = json.loads(result_std)
        return json.dumps(parsed_std) # TODO: print both