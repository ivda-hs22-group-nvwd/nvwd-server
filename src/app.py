import sqlite3
import json
from flask import Flask, render_template, request, jsonify, Response

### INIT TODO:Refactor
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

continuous_cols = [
        'age',
        'height']
categorical_cols = [
        'body_type', 'drinks', 'drugs', 'income', 'job', 'orientation', 'sex', 'smokes',
        'diet','diet_modifier',
        'education_status', 'education_institution',
        'offspring_status', 'offspring_future',
        'pets_cats', 'pets_dogs',
        'religion_type', 'religion_modifier',
        'sign', 'sign_modifier']
ethnities_cols = [
        'ethnicities_middle_eastern', 'ethnicities_hispanic_/_latin',
        'ethnicities_white', 'ethnicities_indian', 'ethnicities_other',
        'ethnicities_asian', 'ethnicities_black', 'ethnicities_native_american',
        'ethnicities_pacific_islander']
speaks_cols = [
        'speaks_english', 'speaks_spanish', 'speaks_french', 'speaks_c++',
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
    
    
@app.route('/api/dev/list',methods = ['POST'])
def get_by_indices():
    if request.method == 'POST':
        data = request.json
        ids = tuple(data['ids'])
        response_dict = {}
        with sqlite3.connect('okcupid.sqlite') as conn:
            cursor = conn.cursor()
            # https://stackoverflow.com/questions/9522971/is-it-possible-to-use-index-as-a-column-name-in-sqlite
            # TODO: Fix [index] (rename)
            cursor.execute(f'SELECT * FROM okcupid_std WHERE [index] in {format(ids)}')
            std = cursor.fetchall()
            response_dict['std'] = std
            
            cursor.execute(f'SELECT * FROM okcupid_clean WHERE [index] in {format(ids)}')
            clean = cursor.fetchall()
            response_dict['clean'] = clean
            
            
            # Get col names
            names = [description[0] for description in cursor.description]

            
            r = []
            for index, (clean_id, std_id) in enumerate(zip(clean, std)):
                clean = dict(zip(names, clean_id))
                std = dict(zip(names, std_id))
                d = {'clean':clean, 'std':std}        
                r.append(d)
            
        conn.close()
        response = json.dumps(r)
        return response
    

@app.route('/api/dev/std',methods = ['POST'])
def get_standarization():
    if request.method == 'POST':
        data = request.json
        sample = pd.DataFrame.from_records(data=[data])
        std = np.round(mapper.transform(sample), 2)
        #return Response(std.to_json(orient="records"), mimetype='application/json')
        
        global df_clean
        #print(df_clean)
        
        global df_std
        #print(df_std)
        #print(std)
        
        df_std = df_std.append(std) # add new input as last row
        lables = cosine_similarity(df_std)[-1] # calucalte cosine similarty and extract last row
        
        # Generate lables
        SIMILARITY_THRESHOLD = 0.8
        #lables = ['not similar' if x < SIMILARITY_THRESHOLD else "similar" for x in lables]
        lables = [0 if x < SIMILARITY_THRESHOLD else 1 for x in lables]
        df_std['lables'] = lables
        
        df = df_std.copy()
        print(df)
        
        # PCA
        PCA_COMPONENTS = 4
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(df)
        scores_pca = pca.transform(df)
        
        
        # KMEANS
        OPTIMAL_N_CLUSTER = 4
        kmeans_pca = KMeans(n_clusters=OPTIMAL_N_CLUSTER, init='k-means++', random_state=420)
        kmeans_pca.fit(scores_pca)
        
        df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
        df_segm_pca_kmeans.columns.values[-PCA_COMPONENTS:] = ['PComp 1', 'PComp 2', 'PComp 3', 'PComp 4']

        df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
        
        df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment K-means PCA'].map({
            0: 'first',
            1: 'second',
            2: 'third',
            3: 'fourth'
        })
        
        # DEBUG
        #return render_template('table.html',  tables=[df_segm_pca_kmeans.to_html(classes='data')], titles=df_segm_pca_kmeans.columns.values)
        
        # WORKS but maybe better method
        #return json.loads(json.dumps(list(df_segm_pca_kmeans.T.to_dict().values())))
        
        return json.dumps(df_segm_pca_kmeans.to_dict(orient='records'), indent=2)
    
@app.route('/api/dev/std/template', methods=("POST", "GET"))
def html_table():
    if request.method == 'POST':
        data = request.json
        sample = pd.DataFrame.from_records(data=[data])
        std = np.round(mapper.transform(sample), 2)
        return render_template('table.html',  tables=[std.to_html(classes='data')], titles=std.columns.values)
    
    
    
@app.route('/api/dev/std/db',methods = ['POST'])
def db_user():
    if request.method == 'POST':
        # USER DATA
        data = request.json
        sample = pd.DataFrame.from_records(data=[data])
        
        
        # DB DATA
        with sqlite3.connect('okcupid.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM okcupid_clean')
            clean = cursor.fetchall()
            col_names_clean = [description[0] for description in cursor.description]
            
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM okcupid_std')
            std = cursor.fetchall()
            
            # col names
            col_names_std = [description[0] for description in cursor.description]
        
        ## df conversion
        df_clean = pd.DataFrame(clean, columns=col_names_clean)
        #df_std = pd.DataFrame(std, columns=col_names_std) # TESTING ONLY
        
        #print(df_clean) # [842 rows x 89 columns] with index
        #print(sample) # [1 rows x 88 columns], no index
        #print(len(col_names_clean)) # [89] with index
        #print(len(col_names_std)) # [89] with index
        
        # MAPPER
        global continuous_cols, categorical_cols, ethnities_cols, speaks_cols
        global mapper
        df_std = np.round(mapper.fit_transform(df_clean.copy()),2)
        
        # HANDLE INPUT
        std = np.round(mapper.transform(sample), 2)

        
        
        # LOGIC
        df_std = df_std.append(std) # add new input as last row
        lables = cosine_similarity(df_std)[-1] # calucalte cosine similarty and extract last row
        
        # Generate lables
        SIMILARITY_THRESHOLD = 0.8
        #lables = ['not similar' if x < SIMILARITY_THRESHOLD else "similar" for x in lables]
        lables = [0 if x < SIMILARITY_THRESHOLD else 1 for x in lables]
        df_std['lables'] = lables
        
        df = df_std.copy()
        
        # PCA
        PCA_COMPONENTS = 4
        pca = PCA(n_components=PCA_COMPONENTS)
        pca.fit(df)
        scores_pca = pca.transform(df)
        
        
        # KMEANS
        OPTIMAL_N_CLUSTER = 4
        kmeans_pca = KMeans(n_clusters=OPTIMAL_N_CLUSTER, init='k-means++', random_state=420)
        kmeans_pca.fit(scores_pca)
        
        df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(scores_pca)], axis=1)
        df_segm_pca_kmeans.columns.values[-PCA_COMPONENTS:] = ['PComp 1', 'PComp 2', 'PComp 3', 'PComp 4']

        df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
        
        df_segm_pca_kmeans['Segment'] = df_segm_pca_kmeans['Segment K-means PCA'].map({
            0: 'first',
            1: 'second',
            2: 'third',
            3: 'fourth'
        })
        
        
        df_clean['PComp 1'] = df_segm_pca_kmeans['PComp 1']
        df_clean['PComp 2'] = df_segm_pca_kmeans['PComp 2']
        df_clean['PComp 3'] = df_segm_pca_kmeans['PComp 3']
        df_clean['PComp 4'] = df_segm_pca_kmeans['PComp 4']
        df_clean['Segment'] = df_segm_pca_kmeans['Segment']
        df_clean['Segment K-means PCA'] = df_segm_pca_kmeans['Segment K-means PCA']
        
        
        # DEBUG
        #return render_template('table.html',  tables=[df_segm_pca_kmeans.to_html(classes='data')], titles=df_segm_pca_kmeans.columns.values)
        
        # WORKS but maybe better method
        #return json.loads(json.dumps(list(df_segm_pca_kmeans.T.to_dict().values())))
        
        return json.dumps(df_clean.to_dict(orient='records'), indent=2)