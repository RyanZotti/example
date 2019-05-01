import functools
import numpy as np
import os
import pandas as pd
import psycopg2
import psycopg2.extras
import shutil
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, SCORERS
from tpot import TPOTRegressor


from functions import *


# A single place for all connection related details
# Storing a password in plain text is bad, but this is for a temp db with default credentials
def connect_to_postgres(host='localhost'):
    connection_string = "host='{host}' dbname='predata_tourism' user='postgres' password='postgres' port=5432".format(host=host)
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
    return connection, cursor


# Create a Pandas DataFrame given a SQL query
def get_df(sql,host='localhost'):
    connection, cursor = connect_to_postgres(host=host)
    cursor = connection.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    connection.close()
    return df


# This isn't really necessary for the Predata assessment
def imputations(df,predictors,target):
    df[predictors] = df[predictors].astype(str)
    for predictor in predictors:
        df.loc[(df[predictor]=='idk'),predictor] = np.NaN
    df[[target]] = df[[target]].astype(str)
    df.loc[(df[target]=='-'),target] = np.NaN
    df[target] = df[target].astype(float)
    medians = df[predictors].median()
    df = df[df[target].notnull()].copy()
    df[[target]] = df[[target]].astype(float)
    df[predictors] = df[predictors].astype(float)
    for predictor in predictors:
        df.loc[(np.isfinite(df[predictor])==False),predictor] = medians[predictor]
    return df


# Returns new model id by auto-incrementing previous models
# by target variable
def create_new_model_id(target):
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT MAX(COALESCE(model_id,0)) AS lastest_model_id
      FROM model_storage
     WHERE target = '{target}'
    """.format(target=target)
    cursor.execute(sql_query)
    row = cursor.fetchone()
    new_model_id = 1  # default if none exist yet
    lastest_model_id = row['lastest_model_id']
    if lastest_model_id is not None:
        new_model_id = lastest_model_id + 1
    cursor.close()
    connection.close()
    return new_model_id


# Used to clean up temporary files
def remove_file(file_path):
    # Delete only if file exists
    if os.path.exists(file_path):
        os.remove(file_path)


# Make directory if not exists
def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Delete directory if exists
def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)


# Save TPOT source
def save_tpot_soure(tpot, target):
    project_path = os.getcwd()
    source_dir = os.path.join(project_path, 'training')
    source_file = 'train_{target}.py'.format(target=target)
    make_directory(source_dir)
    file_path = functools.reduce(
        os.path.join, [source_dir, source_file])
    tpot.export(file_path)


# Save model to Postgres for long-term storage
def save_model_binary_to_postgres(target, pipeline):
    connection, cursor = connect_to_postgres()
    model_id = create_new_model_id(target)
    
    # The joblib tool only exports to file not string
    # so I have to save model to disk, read the file
    # as a binary string, and then save the binary
    # string to Postgres
    
    # Create file path to save the model
    project_path = os.getcwd()
    temp_path = functools.reduce(
        os.path.join,
        [project_path,
         '{0}-pg-write-temp'.format(target)])
    
    # Save the model to disk
    joblib.dump(pipeline, temp_path)
    
    with open(temp_path, 'rb') as f:
        model_binary = f.read()
    
    # Save to binary to Postgres
    sql_insert = """
     BEGIN;
    INSERT INTO model_storage(target,
           model_id,
           model_binary)
    VALUES ('{target}',
           {model_id},
           {model_binary});
    COMMIT;
    """.format(
        target=target,
        model_id=model_id,
        model_binary=psycopg2.Binary(model_binary))
    cursor.execute(sql_insert)
    cursor.close()
    connection.close()
    
    # Remove temp file
    remove_file(temp_path)
    
    return model_id


# Trains, evaluates, and deploys a model
def run_pipeline(data, predictors, target, save_model_source=False):
    # Remove records with missing data
    # TODO: Remove outliers
    clean_data = imputations(data, predictors, target)
    
    # Sklearn has negative losses because its api tries to maximize everything
    # TODO: Parameterize loss function
    loss_function = 'neg_mean_absolute_error'
    
    # Set genetic algorithm parameters
    tpot = TPOTRegressor(generations=10,
                         population_size=20,
                         verbosity=2,
                         scoring=loss_function)
    
    # Separate the train and validation data
    X_train, X_test, y_train, y_test = train_test_split(
        clean_data,
        clean_data[target],
        test_size=0.25,
        random_state=13)
    
    # Start fully-automated machine learning workflow
    tpot.fit(X_train[predictors], y_train)
    
    # Extract best model
    pipeline = tpot.fitted_pipeline_
    
    # Print results of best model
    train_loss = SCORERS[loss_function](
        tpot.fitted_pipeline_,
        X_train[predictors],
        y_train)
    validation_loss = SCORERS[loss_function](
        tpot.fitted_pipeline_,
        X_test[predictors],
        y_test)
    
    print(validation_loss)
    
    # TODO: Add checkpoint here to skip below code if accuracy not sufficient
    
    # Optionally, save final model source code (vs just binaries)
    if save_model_source:
        save_tpot_soure(tpot, target)
    
    # Save model to Postgres
    model_id = save_model_binary_to_postgres(target, pipeline)
    
    # Save basic metadata, like create date
    save_model_meta(target=target, model_id=model_id)
    
    # Save predictors to Postgres for deployment use later
    save_predictors_to_postgres(target, model_id, predictors)
    
    # Store records for QA check during deployment
    store_record_ids(
        target=target,
        model_id=model_id,
        purpose='train',
        record_ids=list(X_train['observation_id'].values))
    store_record_ids(
        target=target,
        model_id=model_id,
        purpose='validation',
        record_ids=list(X_test['observation_id'].values))
    
    # Store loss values for reference check during model deployment
    store_fit_performance(
        target=target,
        model_id=model_id,
        purpose='train',
        loss_function=loss_function,
        performance=train_loss)
    store_fit_performance(
        target=target,
        model_id=model_id,
        purpose='validation',
        loss_function=loss_function,
        performance=validation_loss)


# Load model from local disk
def load_model_file(path):
    model = joblib.load(path)
    return model


# Load model from permanent Postgres storage location
def load_postgres_model(target, model_id):
    # The joblib tool only reads from file not
    # string, so I have to read the string from
    # Postgres, then write it to disk for joblib
    
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT model_binary
      FROM model_storage
     WHERE target = '{target}'
           AND model_id = {model_id}
    """.format(
        target=target,
        model_id=model_id)
    cursor.execute(sql_query)
    model_binary = cursor.fetchone()['model_binary']
    
    # Create file path to save the intermediate model format
    project_path = os.getcwd()
    temp_path = functools.reduce(
        os.path.join,
        [project_path,
         '{0}-pg-read-temp'.format(target)])
    
    # Remove any stale temp models
    remove_file(temp_path)
    
    # Write model temp file in joblib-readable format
    with open(temp_path, 'wb') as f:
        f.write(model_binary)
    
    # Restore model from temp file
    model_restored = load_model_file(temp_path)
    
    # Clean up new temp file
    remove_file(temp_path)
    
    return model_restored


# TODO: Get best model instead of just the most recently trained
def restore_best_model(target):
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT MAX(model_id) AS model_id
      FROM model_storage
     WHERE target = '{target}'
    """.format(
        target=target)
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    if len(rows) > 0:
        model_id = rows[0]['model_id']
        model_restored = load_postgres_model(target,model_id)
        return model_id, model_restored
    else:
        return None  # If no models exist


# Save predictors so that rest API knows which to use and discard
def save_predictors_to_postgres(target, model_id, predictors):
    connection, cursor = connect_to_postgres()
    sql_query = '''
    BEGIN;
    INSERT INTO predictors(target,
           model_id,
           predictor_index,
           predictor)
    VALUES ('{target}',
            {model_id},
            {predictor_index},
           '{predictor}');
    COMMIT;
    '''
    for predictor_index, predictor in enumerate(predictors):
        cursor.execute(sql_query.format(
            target=target,
            model_id=model_id,
            predictor_index=predictor_index,
            predictor=predictor))
    cursor.close()
    connection.close()


# Get predictor names associated with particular model
def get_predictor_names(target, model_id):
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT predictor
      FROM predictors
     WHERE target = '{target}'
           AND model_id = {model_id}
    ORDER BY predictor_index
    """.format(
        target=target,
        model_id=model_id)
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    predictors = []
    for row in rows:
        predictor = row['predictor'].strip()
        predictors.append(predictor)
    cursor.close()
    connection.close()
    return predictors


# Used to replicate model results
def store_record_ids(target,model_id,purpose,record_ids):
    connection, cursor = connect_to_postgres()
    sql_query = '''
    BEGIN;
    INSERT INTO record_ids(
           target,
           model_id,
           purpose,
           observation_id)
    VALUES ('{target}',
            {model_id},
           '{purpose}',
           '{observation_id}');
    COMMIT;
    '''
    for record_id in record_ids:
        cursor.execute(sql_query.format(
            target=target,
            model_id=model_id,
            purpose=purpose,
            observation_id=record_id))
    cursor.close()
    connection.close()


# Store loss values for reference checks during model deployment
def store_fit_performance(target,model_id,purpose,loss_function,performance):
    connection, cursor = connect_to_postgres()
    sql_query = '''
    BEGIN;
    INSERT INTO fit_performance(
           target,
           model_id,
           purpose,
           loss_function,
           performance)
    VALUES ('{target}',
            {model_id},
           '{purpose}',
           '{loss_function}',
           {performance});
    COMMIT;
    '''
    cursor.execute(sql_query.format(
        target=target,
        model_id=model_id,
        purpose=purpose,
        loss_function=loss_function,
        performance=performance))
    cursor.close()
    connection.close()


# Create SQL based on model's specific target and predictor needs
def make_qa_record_query_string(series_id,forecast_horizon,target,model_id,purpose):
    target_name = series_id + '_' + str(forecast_horizon)
    sql_make_inputs = '''

           DROP TABLE IF EXISTS features;

         CREATE TEMP TABLE features
             AS
        (SELECT time_index AS observation_id,
                tourism AS {target_name},
                LAG (tourism, 1 + {forecast_horizon}) OVER lifetime AS lag_1,
                LAG (tourism, 2 + {forecast_horizon}) OVER lifetime AS lag_2,
                LAG (tourism, 3 + {forecast_horizon}) OVER lifetime AS lag_3,
                LAG (tourism, 4 + {forecast_horizon}) OVER lifetime AS lag_4,
                LAG (tourism, 5 + {forecast_horizon}) OVER lifetime AS lag_5,
                LAG (tourism, 6 + {forecast_horizon}) OVER lifetime AS lag_6,
                LAG (tourism, 7 + {forecast_horizon}) OVER lifetime AS lag_7,
                LAG (tourism, 8 + {forecast_horizon}) OVER lifetime AS lag_8,
                LAG (tourism, 9 + {forecast_horizon}) OVER lifetime AS lag_9,
                LAG (tourism, 10 + {forecast_horizon}) OVER lifetime AS lag_10,
                LAG (tourism, 11 + {forecast_horizon}) OVER lifetime AS lag_11,
                LAG (tourism, 12 + {forecast_horizon}) OVER lifetime AS lag_12,
                CASE WHEN time_index % 12 = 0 THEN 1 ELSE 0 END AS month_0,
                CASE WHEN time_index % 12 = 1 THEN 1 ELSE 0 END AS month_1,
                CASE WHEN time_index % 12 = 2 THEN 1 ELSE 0 END AS month_2,
                CASE WHEN time_index % 12 = 3 THEN 1 ELSE 0 END AS month_3,
                CASE WHEN time_index % 12 = 4 THEN 1 ELSE 0 END AS month_4,
                CASE WHEN time_index % 12 = 5 THEN 1 ELSE 0 END AS month_5,
                CASE WHEN time_index % 12 = 6 THEN 1 ELSE 0 END AS month_6,
                CASE WHEN time_index % 12 = 7 THEN 1 ELSE 0 END AS month_7,
                CASE WHEN time_index % 12 = 8 THEN 1 ELSE 0 END AS month_8,
                CASE WHEN time_index % 12 = 9 THEN 1 ELSE 0 END AS month_9,
                CASE WHEN time_index % 12 = 10 THEN 1 ELSE 0 END AS month_10,
                CASE WHEN time_index % 12 = 11 THEN 1 ELSE 0 END AS month_11,
                COUNT(*) OVER (PARTITION BY series_id,
                         sample_type
                         ORDER BY time_index ASC
                         ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
                         AS future_record_count
           FROM time_series
          WHERE LOWER(series_id) = '{series_id}'
                AND LOWER(sample_type) = 'in'
         WINDOW lifetime AS (PARTITION BY series_id,
                sample_type
                ORDER BY time_index ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW));
         '''
    
    sql_get_inputs = '''

         SELECT {predictors_str}, {target}
           FROM features
           JOIN record_ids
                ON features.observation_id = record_ids.observation_id
          WHERE record_ids.target = '{target}'
                AND record_ids.model_id = {model_id}
                AND LOWER(record_ids.purpose) = LOWER('{purpose}')
        '''
    
    predictor_names = get_predictor_names(target, model_id)
    predictors_str = ''
    for predictor_name in predictor_names:
        predictors_str += predictor_name + ','
    predictors_str = predictors_str[:-1]  # remove trailing comma
    
    sql_merged = sql_make_inputs + sql_get_inputs
    sql_merged = sql_merged.format(
        series_id=series_id,
        target_name=target_name,
        forecast_horizon=forecast_horizon,
        predictors_str=predictors_str,
        purpose=purpose,
        model_id=model_id,
        target=target)
    
    return sql_merged


# Look up how well model performed on a particular set of data
def get_model_fit_performance(target,model_id,purpose,loss_function):
    connection, cursor = connect_to_postgres()
    sql_query = '''
    SELECT performance
      FROM fit_performance
     WHERE target = '{target}'
           AND model_id = {model_id}
           AND LOWER(purpose) = LOWER('{purpose}')
           AND LOWER(loss_function) = LOWER('{loss_function}')
    '''.format(
        target=target,
        model_id=model_id,
        purpose=purpose,
        loss_function=loss_function)
    cursor.execute(sql_query)
    performance = cursor.fetchone()[0]
    return performance


# Save basic master data about a model
def save_model_meta(target,model_id):
    connection, cursor = connect_to_postgres()
    sql_query = '''
    BEGIN;
    INSERT INTO models(
           target,
           model_id,
           create_date)
    VALUES ('{target}',
            {model_id},
            NOW()::DATE);
    COMMIT;
    '''
    cursor.execute(sql_query.format(
        target=target,
        model_id=model_id))
    cursor.close()
    connection.close()


# Used to check that you pulled the model that you expected
def evaluate_fit(series_id,forecast_horizon,target, model_id, model, purpose, loss_function):
    # Get score saved from model training
    offline_score = get_model_fit_performance(
        target=target,
        model_id=model_id,
        purpose=purpose,
        loss_function=loss_function)
    
    # Pull exact records used during training
    sql_query = make_qa_record_query_string(
        series_id=series_id,
        forecast_horizon=forecast_horizon,
        target=target,
        model_id=model_id,
        purpose=purpose)
    records = get_df(sql_query)
    
    # Get ordered predictor names
    predictor_names = get_predictor_names(
        target=target,
        model_id=model_id)
    
    # Calculate loss model being tested
    new_score = SCORERS[loss_function](
        model,
        records[predictor_names],
        records[target])
    
    # Fail if the two scores are not practically identical
    diff = abs(offline_score - new_score)
    assert (diff < 0.0001)


def get_batch_score(target, model_id, data):
    # Get ordered predictor names
    predictor_names = get_predictor_names(
        target=target,
        model_id=model_id)
    
    # Load model
    model = load_postgres_model(target, model_id)
    
    # Make and return predictions
    scores = model.predict(data[predictor_names])
    return scores