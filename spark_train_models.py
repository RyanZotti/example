from pyspark import SparkContext

from functions import *
from data_config import create_data_prep_table_sql, predictors, pull_data_sql


def train_model(target_name):
    series_id, forecast_horizon = target_name.split('_')
    
    # Get training data
    connection, cursor = connect_to_postgres()
    cursor.execute(create_data_prep_table_sql.format(
        forecast_horizon=forecast_horizon,
        target_name=target_name,
        series_id=series_id))
    cursor.close()
    connection.close()
    df = get_df(pull_data_sql.format(target_name=target_name, series_id=series_id))
    
    # Run model training and evaluation pipeline
    print('Start training for {target} ... {target_index} of {total}'.format(
        target=target_name,
        target_index=forecast_horizon,
        total=len(targets)))
    print('Target: {0}'.format(target_name))
    run_pipeline(df, predictors, target_name)
    print('Finished training {target}'.format(target=target_name))


sc = SparkContext()
targets = []
for series_index in range(1, 367):
    series_id = 'm'+str(series_index)
    for horizon in range(1, 25):
        target = '{series_id}_{forecast_horizon}'.format(
            series_id=series_id,
            forecast_horizon=horizon)
        targets.append(target)

rdd_targets = sc.parallelize(targets)
rdd_targets.map(train_model).collect()
print('Finished successfully.')