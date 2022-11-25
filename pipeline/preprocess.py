# IMPORT
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# CONST
ZODIAC_STRING_REPLACMENT = '&rsquo;' # corresponds to " ' "
OFFSPRING_STRING_REPLACMENT = '&rsquo;' # corresponds to " ' "

# FUNCTIONS
# Using standard scaler
def std_scaler(df, col_names):
    scaled_features = df.copy()
 
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
 
    scaled_features[col_names] = features

    return scaled_features


# Using min/max scaler
def minmax_scaler(df, col_names):
    scaled_features = df.copy()
 
    features = scaled_features[col_names]
    scaler = MinMaxScaler().fit(features.values)
    features = scaler.transform(features.values)
 
    scaled_features[col_names] = features

    return scaled_features

def preprocess(columns, df):
    MAX_AGE = 105
    MISSING_DIET_MODIFIER = 'No specified diet modifier'
    MISSING_SIGN_MODIFIER = 'No specified sign modifier'
    
    #for column in df:
     #   if column in columns:
      #      pass
       # else:
        #    df.drop(column, axis=1)


    if 'age' in columns:
        ### AGE ###

        # Remove nan's
        df.dropna(inplace=True, subset=['age'])
        
        df.drop(df[df['age'] >= MAX_AGE].index, inplace = True)



    if 'body_type' in columns:
        ### BODY_TYPE ###
        # Remove nan's

        df.dropna(inplace=True, subset=['body_type'])
        
        """
        # Encode body type
        body_type_encoder = LabelEncoder()
        body_type_encoder.fit(df['body_type'])
        encoded_col_body_type = body_type_encoder.transform(df['body_type'])
        df['body_type'] = encoded_col_body_type
        """

    
    if 'diet' in columns:
        ### DIET ###

        # Remove nan's
        df.dropna(inplace=True, subset=['diet'])

        # Extract diet modifier
        df['diet_modifier'] = df['diet'].str.split(' ').str[:-1]
        df['diet_modifier'] = df['diet_modifier'].apply(lambda y: MISSING_DIET_MODIFIER if len(y)==0 else y[0]) # replace empty lists with MISSING_DIET_MODIFIER' and extract term from list
        
        # Extract only diet
        df['diet'] = df['diet'].str.split(' ').str[-1]

        """ 
        # Encode diet
        diet_encoder = LabelEncoder()
        diet_encoder.fit(df['diet'])
        encoded_col_diet = diet_encoder.transform(df['diet'])
        df['diet'] = encoded_col_diet

        # Encode diet modifier
        diet_modifier_encoder = LabelEncoder()
        diet_modifier_encoder.fit(df['diet_modifier'])
        encoded_col_diet_modifier = diet_modifier_encoder.transform(df['diet_modifier'])
        df['diet_modifier'] = encoded_col_diet_modifier
        """


    if 'drinks' in columns:
        ### DRINKS ###

        # Remove nan's
        df.dropna(inplace=True, subset=['drinks'])

        """
        # Encode drinks modifier
        drinks_encoder = LabelEncoder()
        drinks_encoder.fit(df['drinks'])
        encoded_col_drinks = drinks_encoder.transform(df['drinks'])
        df['drinks'] = encoded_col_drinks
        """


    if 'drugs' in columns:
        ### DRUGS ###

        # Remove nan's
        df.dropna(inplace=True, subset=['drugs'])

        """
        # Encode drugs modifier
        drinks_encoder = LabelEncoder()
        drinks_encoder.fit(df['drugs'])
        encoded_col_drugs = drinks_encoder.transform(df['drugs'])
        df['drugs'] = encoded_col_drugs
        """


    if 'education' in columns:
        ### EDUCATION ###

        # Remove nan's
        df.dropna(inplace=True, subset=['education'])


        # Extract only education institution
        # Todo find better solution to use the dedicated mapper in naming.yaml
        def education_institution_mapper(x):
            if 'college/university' in x:
                return 'college/university'
            if 'two-year college' in x:
                return 'two-year college'
            if 'masters program' in x:
                return 'masters program'
            if 'ph.d program' in x:
                return 'ph.d program'
            if 'high school' in x:
                return 'high school'
            if 'law school' in x:
                return 'law school'
            if 'med school' in x:
                return 'med school'
            if 'space camp' in x:
                return 'space camp'

        # Extract only education status
        def education_status_mapper(x):
            if 'dropped out of' in x:
                return 'dropped out of'
            if 'working on' in x:
                return 'working on'
            if 'graduated from' in x:
                return 'graduated from'


        df['education_status'] = df['education'].apply(lambda x: education_status_mapper(x))
        df['education_institution'] = df['education'].apply(lambda x: education_institution_mapper(x))

        """
        # Encode education_status
        education_status_encoder = LabelEncoder()
        education_status_encoder.fit(df['education_status_extracted'])
        encoded_col_education_status = education_status_encoder.transform(df['education_status_extracted'])
        df['education_status_extracted'] = encoded_col_education_status

        # Encode diet modifier
        education_institution_encoder = LabelEncoder()
        education_institution_encoder.fit(df['education_institution_extracted'])
        encoded_col_education_institution = education_institution_encoder.transform(df['education_institution_extracted'])
        df['education_institution_extracted'] = encoded_col_education_institution
        """

        # Drop reduandant cols
        df = df.drop('education', axis=1)


    if 'ethnicity' in columns:
        ### ETHNICITY ###

        # Extract all ethnicities categories
        # Get all distinct values for the ethnicity  col
        ethnicities = df.ethnicity.unique()

        # Clean
        ethnicities = [e for e in ethnicities if str(e) != 'nan'] # remove nan values

        # Extract all ethnicities combinations 
        ethnicities = ', '.join(ethnicities)
        ethnicities = ethnicities.split(', ') 
        ethnicities = [*set(ethnicities)] # create list of "base" ethnicities

        # Generate new header for encoded categories
        ethnicities_encoded_header = ['ethnicities_{}'.format(e.replace(' ', '_')) for e in ethnicities]


        # Remove nan's
        df.dropna(inplace=True, subset=['ethnicity'])

        
        # Add col header
        for eth_col in ethnicities_encoded_header:
            df[eth_col] = np.nan

        # Filter
        def filter_ethnicities(col, row_ethnicities):
            # extract all ethnicities from the col 'ethnicity'
            row_ethnicities = row_ethnicities.split(', ')
            
            # compare all extracted to current row in df
            for re in row_ethnicities:
                # match
                if re == col:
                    return 1
            # no match
            return 0

        
        # Hot encoding for all ethnicities cols
        for (ethnicities_encoded_header_col, e) in zip(ethnicities_encoded_header, ethnicities):
            df[ethnicities_encoded_header_col] = df.apply(lambda x: filter_ethnicities(e, x['ethnicity']), axis=1)
        

        # Drop reduandant cols
        df = df.drop('ethnicity', axis=1)


    if 'height' in columns:
        ### HEIGHT ###

        # Remove nan's
        df.dropna(inplace=True, subset=['height'])

        """
        # Scale
        df = std_scaler(df, ['height'])
        """


    if 'income' in columns:
        ###  ###

        # Replace -1 entries
        df['income'] = df['income'].apply(lambda y: np.nan if y==-1 else y) # replace -1 with nan

        # Remove nan's
        df.dropna(inplace=True, subset=['income'])

        """
        # Encode income
        income_encoder = LabelEncoder()
        income_encoder.fit(df['income'])
        encoded_col_income = income_encoder.transform(df['income'])
        df['income'] = encoded_col_income
        """


    if 'job' in columns:
        ### JOB ###

        # Remove nan's
        df.dropna(inplace=True, subset=['job'])

        """
        # Encode job
        job_encoder = LabelEncoder()
        job_encoder.fit(df['job'])
        encoded_col_job = job_encoder.transform(df['job'])
        df['job'] = encoded_col_job
        """


    if 'offspring' in columns:
        ### OFFSPRING  ###

        # Extract all offspring categories
        # todo: automate

        OFFSPRING_STATUS_ORIG = [
            'doesn\'t have kids', 'has a kid', 'has kids'] # STATUS


        OFFSPRING_FUTURE_ORIG = [
            'and doesn\'t want any', 'doesn\'t want kids', 'but doesn\'t want more',
            'but might want them', 'might want kids', 'and might want more',
            'wants kids', 'but wants them', 'and wants more'] # FUTURE

        OFFSPRING_FUTURE = [
            'doesn\'t want',
            'might want',
            'wants'
        ]

        # Remove nan's
        df.dropna(inplace=True, subset=['offspring'])

        df['offspring'] = df['offspring'].str.replace(OFFSPRING_STRING_REPLACMENT,'\'')  # replace 

        offspring_encoded_header = ['offspring_status', 'offspring_future']

        # Add col header
        for off_col in offspring_encoded_header:
            df[off_col] = np.nan

        # Filer
        def filter_offspring_status(row_offspring):    
            # compare all extracted to current row in df
            for status in OFFSPRING_STATUS_ORIG:
                if status in row_offspring:
                    # match
                    return status
            # no match
            return np.nan

        # Filter
        def filter_offspring_future(row_offspring):    
            # compare all extracted to current row in df
            for future in OFFSPRING_FUTURE:
                if future in row_offspring:
                    # match
                    return future
            # no match
            return np.nan

        # Hot encoding for both offspring cols
        df['offspring_status'] = df.apply(lambda x: filter_offspring_status(x['offspring']), axis=1)
        df['offspring_future'] = df.apply(lambda x: filter_offspring_future(x['offspring']), axis=1)

        df.dropna(inplace=True, subset=['offspring_status'])
        df.dropna(inplace=True, subset=['offspring_future'])


        """
        # Encode offspring_status
        offspring_status_encoder = LabelEncoder()
        offspring_status_encoder.fit(df['offspring_status'])
        encoded_col_offspring_status = offspring_status_encoder.transform(df['offspring_status'])
        df['offspring_status'] = encoded_col_offspring_status

        # Encode offspring_future
        offspring_future_encoder = LabelEncoder()
        offspring_future_encoder.fit(df['offspring_future'])
        encoded_col_offspring_future = offspring_future_encoder.transform(df['offspring_future'])
        df['offspring_future'] = encoded_col_offspring_future
        """


        # Drop reduandant cols
        df = df.drop('offspring', axis=1)


    if 'orientation' in columns:
        ### ORIENTATION ###

        # Remove nan's
        df.dropna(inplace=True, subset=['orientation'])

        """
        # Encode orientation
        orientation_encoder = LabelEncoder()
        orientation_encoder.fit(df['orientation'])
        encoded_col_orientation = orientation_encoder.transform(df['orientation'])
        df['orientation'] = encoded_col_orientation
        """


    if 'pets' in columns:
        ### PETS ###

        # Extract all pets categories
        # todo: automate

        PETS_CATS = [
            'has cats', 'likes cats', 'dislikes cats']

        PETS_DOGS = [
            'has dogs', 'likes dogs', 'dislikes dogs']

        # Remove nan's
        df.dropna(inplace=True, subset=['pets'])


        pets_encoded_header = ['pets_cats', 'pets_dogs']

        # Add col header
        for pets_col in pets_encoded_header:
            df[pets_col] = np.nan

        # Filer
        def filter_pets_cats(row_pets):    
            # compare all extracted to current row in df
            for relation in PETS_CATS:
                if relation in row_pets:
                    # match
                    return relation
            # no match
            return np.nan

        # Filer
        def filter_pets_dogs(row_pets):    
            # compare all extracted to current row in df
            for relation in PETS_DOGS:
                if relation in row_pets:
                    # match
                    return relation
            # no match
            return np.nan


        # Hot encoding for both offspring cols
        df['pets_cats'] = df.apply(lambda x: filter_pets_cats(x['pets']), axis=1)
        df['pets_dogs'] = df.apply(lambda x: filter_pets_dogs(x['pets']), axis=1)

        df.dropna(inplace=True, subset=['pets_cats'])
        df.dropna(inplace=True, subset=['pets_dogs'])


        """
        # Encode pets_cats
        pets_cats_encoder = LabelEncoder()
        pets_cats_encoder.fit(df['pets_cats'])
        encoded_col_pets_cats = pets_cats_encoder.transform(df['pets_cats'])
        df['pets_cats'] = encoded_col_pets_cats

        # Encode pets_dogs
        pets_dogs_encoder = LabelEncoder()
        pets_dogs_encoder.fit(df['pets_dogs'])
        encoded_col_pets_dogs = pets_dogs_encoder.transform(df['pets_dogs'])
        df['pets_dogs'] = encoded_col_pets_dogs
        """


        # Drop reduandant cols
        df = df.drop('pets', axis=1)


    if 'religion' in columns:
        ### RELIGION ###

        # Extract all offspring categories
        # todo: automate

        # Extract all religion categories
        # Get all distinct values for the religion  col
        religion = df.religion.unique()

        # Clean
        religion = [r for r in religion if str(r) != 'nan'] # remove nan values

        # Extract all religion types
        religion_types = []
        religion_modifiers = [] 
        for r in religion:
            # extraxt first half: up to 'and' or 'but'
            if 'and' in r:
                religion_extracted = r.split('and')[0]
            elif 'but' in r:
                religion_extracted = r.split('but')[0]
            else:
                religion_extracted = r
            religion_types.append(religion_extracted)
        
        for r in religion:
            # extraxt first half: up to 'and' or 'but'
            if 'and' in r:
                religion_modifier_extracted = r.split('and')[1]
            elif 'but' in r:
                religion_modifier_extracted = r.split('but')[1]
            
            religion_modifiers.append(religion_modifier_extracted)


        religion_types = [*set(religion_types)] # create list of "base" religions


        religion_modifiers = [*set(religion_modifiers)] # create list of religion modifiers


        RELIGION_TYPES = religion_types
        RELIGION_MODIFIERS = religion_modifiers

        # Remove nan's
        df.dropna(inplace=True, subset=['religion'])

        relgion_encoded_header = ['religion_type', 'religion_modifier']

        # Add col header
        for rel_col in relgion_encoded_header:
            df[rel_col] = np.nan

        # Filer
        def filter_religion_type(row_religion):    
            # compare all extracted to current row in df
            for type in RELIGION_TYPES:
                if type in row_religion:
                    # match
                    return type
            # no match
            return np.nan

        # Filter
        def filter_religion_modifier(row_religion):    
            # compare all extracted to current row in df
            for relmodifier in RELIGION_MODIFIERS:
                if relmodifier in row_religion:
                    # match
                    return relmodifier
            # no match
            return np.nan

        # Hot encoding for both offspring cols
        df['religion_type'] = df.apply(lambda x: filter_religion_type(x['religion']), axis=1)
        df['religion_modifier'] = df.apply(lambda x: filter_religion_modifier(x['religion']), axis=1)

        
        df.dropna(inplace=True, subset=['religion_type'])
        df.dropna(inplace=True, subset=['religion_modifier'])

        """
        # Encode religion_type
        religion_type_encoder = LabelEncoder()
        religion_type_encoder.fit(df['religion_type'])
        encoded_col_religion_type = religion_type_encoder.transform(df['religion_type'])
        df['religion_type'] = encoded_col_religion_type

        # Encode religion_modifier
        religion_modifier_encoder = LabelEncoder()
        religion_modifier_encoder.fit(df['religion_modifier'])
        encoded_col_religion_modifier = religion_modifier_encoder.transform(df['religion_modifier'])
        df['religion_modifier'] = encoded_col_religion_modifier
        """

        # Drop reduandant cols
        df = df.drop('religion', axis=1)


    if 'sex' in columns:
        ### SEX ###

        # Remove nan's
        df.dropna(inplace=True, subset=['sex'])

        """
        # Encode drugs modifier
        sex_encoder = LabelEncoder()
        sex_encoder.fit(df['sex'])
        encoded_col_sex = sex_encoder.transform(df['sex'])
        df['sex'] = encoded_col_sex
        """


    if 'sign' in columns:
        ### SIGN ###

        # Remove nan's
        df.dropna(inplace=True, subset=['sign'])


        # Extract sign modifier
        df['sign_modifier'] = df['sign'].str.split(' ').str[1:]
        df['sign_modifier'] = df['sign_modifier'].apply(lambda y: MISSING_SIGN_MODIFIER if len(y)==0 else y) # replace empty lists with MISSING_SIGN_MODIFIER
        df['sign_modifier'] = df['sign_modifier'].apply(lambda y: ' '.join(y) if len(y)!=0 else y) # join list of strings together
        df['sign_modifier'] = df['sign_modifier'].str.replace(ZODIAC_STRING_REPLACMENT,'\'')  # replace 
        

        # Extract only sign
        df['sign'] = df['sign'].str.split(' ').str[0]



        
        """
        # Encode sign
        sign_encoder = LabelEncoder()
        sign_encoder.fit(df['sign_extracted'])
        encoded_col_sign = sign_encoder.transform(df['sign_extracted'])
        df['sign_extracted'] = encoded_col_sign

        # Encode sign modifier
        sign_modifier_encoder = LabelEncoder()
        sign_modifier_encoder.fit(df['sign_modifier_extracted'])
        encoded_col_sign_modifier = sign_modifier_encoder.transform(df['sign_modifier_extracted'])
        df['sign_modifier_extracted'] = encoded_col_sign_modifier
        """

        # Drop reduandant cols
        #df = df.drop('sign', axis=1)


    if 'smokes' in columns:
        ### SMOKES ###

        # Remove nan's
        df.dropna(inplace=True, subset=['smokes'])

        """
        # Encode smokes modifier
        smokes_encoder = LabelEncoder()
        smokes_encoder.fit(df['smokes'])
        encoded_col_smokes = smokes_encoder.transform(df['smokes'])
        df['smokes'] = encoded_col_smokes
        """


    if 'speaks' in columns:
        ### SPEAKS ###

        # Remove nan's
        df.dropna(inplace=True, subset=['speaks'])

        languages = df.speaks.unique()

        language = []
        language_level = []

        for l in languages:
            entries = l.split(', ')
            for e in entries:

                # at least on entry that has a modifier
                if e.find('(') != -1:
                    # extract modifier
                    res = e[e.find('(')+1:e.find(')')]
                    
                    # check if modifier can be appended
                    if res not in language_level:
                        language_level.append(res)
                    
                    # check if language can be appended
                    if e[:e.find(' ')]:
                        if e[:e.find(' ')] not in language:
                            language.append(e[:e.find(' ')])
                
                # no modifier
                else:
                    # check if language can be appended
                    if e not in language:
                        language.append(e)


        SPEAKS_LANGUAGE = language
        SPEAKS_LANGUAGE_LEVEL = language_level

        speaks_encoded_header = [l.replace(' ', '_') for l in SPEAKS_LANGUAGE]

        # Add col header
        for speaks_col in speaks_encoded_header:
            df['speaks_'+speaks_col] = np.nan

        speaks_encoded_header = ['speaks_'+l for l in speaks_encoded_header]
        speaks_encoded_header = [l.replace(' ', '_') for l in speaks_encoded_header]


        # Filter
        def filter_speaks(s, row_speaks):    
            # compare all extracted to current row in df

            # split string into list of multiple langues + modifier
            rs = row_speaks.split(', ')

            # check if language s (current col) is in this list
            res = [i for i in rs if s in i]
            if len(res) != 0:
                # modifier:
                if '(fluently)' in res[0]:
                    return 4
                if '(ok)' in res[0]:
                    return 3
                if '(poorly)' in res[0]:
                    return 1
                else:
                    return 2
            else:
                return 0 # maybe change to np.nan


        # Hot encoding for all speaks cols
        for (speaks_encoded_header_col, s) in zip(speaks_encoded_header, SPEAKS_LANGUAGE):
            df[speaks_encoded_header_col] = df.apply(lambda x: filter_speaks(s, x['speaks']), axis=1)


        # Drop reduandant cols
        df = df.drop('speaks', axis=1)

    if 'status' in columns:
        ### STATUS ###

        # Remove nan's
        df.dropna(inplace=True, subset=['status'])

        """
        # Encode drugs modifier
        status_encoder = LabelEncoder()
        status_encoder.fit(df['status'])
        encoded_col_status = status_encoder.transform(df['status'])
        df['status'] = encoded_col_status
        """
    
    return df





def normalize(df):
    print('test')