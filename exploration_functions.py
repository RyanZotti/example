import matplotlib.pyplot as plt
import seaborn as sns

from data_config import *
from functions import *


# Plot in or oos data of a simple time series
def plot_series_sample(series_id, sample_type):
    assert (sample_type in ['in', 'oos'])
    sql_query = """
    SELECT time_index,
           tourism
      FROM time_series
     WHERE sample_type = '{sample_type}'
           AND series_id = '{series_id}'
     ORDER BY time_index ASC
    """.format(
        series_id=series_id,
        sample_type=sample_type)

    df = get_df(sql_query)

    sns.set_style(style='darkgrid')
    fig = plt.figure(figsize=(10, 8))
    title_font = {'fontname': 'Arial', 'size': '16'}
    fig.suptitle('{series_id} - {sample_type}'.format(
        series_id=series_id,
        sample_type=sample_type.upper()), **title_font)

    # Bring title closer to rest of plot
    fig.subplots_adjust(top=0.95)

    # Enlarge axis names
    axis_title_font = {'fontname': 'Arial', 'size': '14'}
    plt.xlabel('Time', **axis_title_font)
    plt.ylabel('Tourism', **axis_title_font)

    # Enlarge tick font size
    axis_tick_fonts = {'fontname': 'Arial', 'size': '12'}
    plt.xticks(**axis_tick_fonts)
    plt.yticks(**axis_tick_fonts)

    # Plot median absolute error
    plt.plot(df['time_index'],
             df['tourism'],
             color='#4c72b0',
             linewidth=3)

    plt.show()


# Gets the denominator of MASE for a particular series ID
def get_scale_factor(series_id):
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT error AS scale_factor
      FROM scale_factor
     WHERE series_id = '{series_id}'
    """.format(series_id=series_id)
    cursor.execute(sql_query)
    row = cursor.fetchone()
    scale_factor = row[0]
    cursor.close()
    connection.close()
    return scale_factor


# General purpose function for getting data
def get_model_inputs(series_id, forecast_horizon, pred_type):
    assert (pred_type in ['validation', 'oos'])

    # forecast_horizon should be 0 in lags but equal to horizon in modulus
    where_clause = ''
    if pred_type.lower() == 'validation':
        if series_id[0].lower() == 'm':
            where_clause = 'WHERE future_record_count = 24'
        elif series_id[0].lower() == 'q':
            where_clause = 'WHERE future_record_count = 8'
        else:
            where_clause = 'WHERE future_record_count = 1'

    target_name = series_id + '_' + str(forecast_horizon)
    sql_make_inputs = '''

       DROP TABLE IF EXISTS features;

     CREATE TEMP TABLE features
         AS
    (SELECT time_index + {forecast_horizon} AS observation_id,
            tourism AS {target_name},
            LAG (tourism, 1) OVER lifetime AS lag_1,
            LAG (tourism, 2) OVER lifetime AS lag_2,
            LAG (tourism, 3) OVER lifetime AS lag_3,
            LAG (tourism, 4) OVER lifetime AS lag_4,
            LAG (tourism, 5) OVER lifetime AS lag_5,
            LAG (tourism, 6) OVER lifetime AS lag_6,
            LAG (tourism, 7) OVER lifetime AS lag_7,
            LAG (tourism, 8) OVER lifetime AS lag_8,
            LAG (tourism, 9) OVER lifetime AS lag_9,
            LAG (tourism, 10) OVER lifetime AS lag_10,
            LAG (tourism, 11) OVER lifetime AS lag_11,
            LAG (tourism, 12) OVER lifetime AS lag_12,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 0 THEN 1 ELSE 0 END AS month_0,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 1 THEN 1 ELSE 0 END AS month_1,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 2 THEN 1 ELSE 0 END AS month_2,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 3 THEN 1 ELSE 0 END AS month_3,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 4 THEN 1 ELSE 0 END AS month_4,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 5 THEN 1 ELSE 0 END AS month_5,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 6 THEN 1 ELSE 0 END AS month_6,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 7 THEN 1 ELSE 0 END AS month_7,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 8 THEN 1 ELSE 0 END AS month_8,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 9 THEN 1 ELSE 0 END AS month_9,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 10 THEN 1 ELSE 0 END AS month_10,
            CASE WHEN (time_index + {forecast_horizon}) % 12 = 11 THEN 1 ELSE 0 END AS month_11,
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

     SELECT *
       FROM features
      {where_clause}
      ORDER BY observation_id DESC
      LIMIT 1
    '''

    # Fill in placeholders specific to the model
    sql_merged = sql_make_inputs + sql_get_inputs
    sql_merged = sql_merged.format(
        series_id=series_id,
        target_name=target_name,
        forecast_horizon=forecast_horizon,
        where_clause=where_clause)

    # Pull and return the data
    data = get_df(sql_merged)
    return data


# Used for time index continuity between in and oos data
def get_max_time(series_id):
    connection, cursor = connect_to_postgres()
    sql_query = """
    SELECT max_time
      FROM max_time
     WHERE series_id = '{series_id}'
    """.format(series_id=series_id)
    cursor.execute(sql_query)
    row = cursor.fetchone()
    max_time = row[0]
    cursor.close()
    connection.close()
    return max_time


# Auto-generates the target variables for particular series ID
def get_targets(series_id):
    if series_id[0].lower() == 'm':
        max_horizon = 24
    elif series_id[0].lower() == 'q':
        max_horizon = 8
    else:
        max_horizon = 1
    targets = []
    for horizon in range(1, max_horizon + 1):
        target = series_id + '_' + str(horizon)
        targets.append(target)
    return targets


# Only used for validation and oos predictions
def get_predictions(series_id, pred_type):
    predictions = []
    targets = get_targets(series_id)
    for horizon, target in enumerate(targets):
        # Get best model
        model_id, model_restored = restore_best_model(target)

        # Get ordered predictor names
        predictor_names = get_predictor_names(
            target=target,
            model_id=model_id)

        # Get model inputs
        model_inputs = get_model_inputs(series_id, horizon, pred_type)

        # Get and record prediction
        prediction = model_restored.predict(model_inputs[predictor_names])[0]
        predictions.append(prediction)
    return predictions


# Because oos vs in are separate data sets I have different logic for pulling their data
def get_oos_data(series_id, phase_shift=False):
    assert (series_id[0].lower() in ['m', 'y', 'q'])

    # Set WHERE clause filter in SQL code
    # Used to ignore or apply phase shift
    if phase_shift is True:
        if series_id[0].lower() == 'm':
            time_restriction = 'time_series.time_index > 2'
        elif series_id[0].lower() == 'm':
            time_restriction = 'time_series.time_index > 2'
        else:
            time_restriction = 'time_series.time_index > 1'
    else:
        if series_id[0].lower() == 'm':
            time_restriction = 'time_series.time_index < 24'
        elif series_id[0].lower() == 'm':
            time_restriction = 'time_series.time_index < 8'
        else:
            time_restriction = 'time_series.time_index < 1'

    sql_oos = '''
    SELECT time_index + max_time.max_time AS time_index,
           tourism
      FROM time_series
      JOIN max_time
           ON time_series.series_id = max_time.series_id
     WHERE time_series.sample_type = 'oos'
           AND time_series.series_id = '{series_id}'
           AND {time_restriction}
     ORDER BY time_index ASC
    '''.format(
        series_id=series_id,
        time_restriction=time_restriction)
    df_oos = get_df(sql_oos)
    return df_oos


# Plots all of a series id's model errors
def plot_horizon_error(series_id):
    if series_id[0].lower() == 'm':
        series_type = 'months'
    elif series_id[0].lower() == 'q':
        series_type = 'quarters'
    else:
        series_type = 'years'

    error_bands = get_error_bounds(series_id)

    sns.set_style(style='darkgrid')
    fig = plt.figure(figsize=(10, 8))
    title_font = {'fontname': 'Arial', 'size': '16'}
    fig.suptitle('{series_id} Forecast Error Over Time'.format(series_id=series_id.upper()), **title_font)

    # Bring title closer to rest of plot
    fig.subplots_adjust(top=0.95)

    # Enlarge axis names
    axis_title_font = {'fontname': 'Arial', 'size': '14'}
    plt.xlabel('Forecast Horizon ({series_type})'.format(series_type=series_type), **axis_title_font)
    plt.ylabel('Mean Absolute Error', **axis_title_font)

    # Enlarge tick font size
    axis_tick_fonts = {'fontname': 'Arial', 'size': '12'}
    plt.xticks(**axis_tick_fonts)
    plt.yticks(**axis_tick_fonts)

    # Draw the plot
    plt.plot(range(1, len(error_bands) + 1),
             error_bands,
             color='#4c72b0',
             linewidth=3)

    plt.xlim(1, len(error_bands))


# Gets MASE for a particular series
def get_mase(series_id, pred_type, phase_shift=False):
    assert (pred_type in ['validation', 'oos'])
    if pred_type == 'validation':
        df = get_validation_data(series_id)
    else:
        df = get_oos_data(series_id, phase_shift)
    predictions = get_predictions(series_id, pred_type)
    df['score'] = predictions
    df['error'] = abs(df['tourism'] - df['score'])
    avg_error = df['error'].mean()
    scale_factor = get_scale_factor(series_id)
    scaled_error = avg_error / scale_factor
    return scaled_error


# Used to get the error bounds for forecast visualization
def get_error_bounds(series_id):
    error_bounds = []
    targets = get_targets(series_id)
    for target in targets:
        model_id, model_restored = restore_best_model(target)
        offline_score = get_model_fit_performance(
            target=target,
            model_id=model_id,
            purpose='validation',
            loss_function='neg_mean_absolute_error')
        error_bounds.append(abs(offline_score))
    return error_bounds


# Validation-equivalent to get_oos_data()
def get_validation_data(series_id):
    where_clause = ''
    if series_id[0].lower() == 'm':
        where_clause = 'future_record_count < 24'
    elif series_id[0].lower() == 'q':
        where_clause = 'future_record_count < 8'
    else:
        where_clause = 'future_record_count < 1'
    df_actual = get_df("""
        CREATE TEMP TABLE data
            AS
        SELECT time_index, tourism,
               COUNT(*) OVER future AS future_record_count
          FROM time_series
         WHERE series_id = '{series_id}'
               AND sample_type = 'in'
        WINDOW future AS (PARTITION BY series_id,
               sample_type
               ORDER BY time_index ASC
               ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING);

        SELECT time_index, tourism
          FROM data
         WHERE {where_clause}
         ORDER BY time_index ASC
    """.format(
        series_id=series_id,
        where_clause=where_clause))
    return df_actual


# Generic function to plot predictions
def plot_predictions(series_id, pred_type, phase_shift=False):
    predictions = get_predictions(series_id, pred_type)

    assert (pred_type in ['validation', 'oos'])
    if pred_type.lower() == 'validation':
        df_actual = get_validation_data(series_id)
    else:
        df_actual = get_oos_data(series_id, phase_shift)

    # TODO: Make this explicit
    # Implicit assumption that orders match betwee the two datasets
    df_actual['score'] = predictions
    error_bands = get_error_bounds(series_id)
    df_actual['error_band'] = error_bands

    colors = ['#DC0A16', '#FD8226', '#FDDA31', '#75A21D', '#42632B']
    sns.set_style(style='darkgrid')

    # Ensure graph and text is large enough to read
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    ax = plt.gca()

    plot_1 = ax.plot(
        df_actual['time_index'],
        df_actual['score'],
        label='Predicted',
        color=colors[4],
        linewidth=2)

    plot_2 = ax.plot(
        df_actual['time_index'],
        df_actual['tourism'],
        label='Actual',
        color=colors[2],
        linewidth=2)

    # Plot upper and lower quartiles
    plt.fill_between(df_actual['time_index'],
                     df_actual['score'] - df_actual['error_band'],
                     df_actual['score'] + df_actual['error_band'],
                     facecolor=colors[4],
                     alpha=0.25,
                     linewidth=0.0)

    ax.set_xlabel('Month(t)')
    ax.set_ylabel('Tourism')
    pred_type_formatted = pred_type
    pred_type_formatted = pred_type[0].upper() + pred_type[1:]
    mase = get_mase(
        series_id=series_id,
        pred_type=pred_type,
        phase_shift=phase_shift)
    mase = str(mase)[:4]
    plt.title("{id} Time Series {pred_type} with MASE = {MASE} and Phase Shift Set to {phase_shift}".format(
        id=series_id.upper(),
        pred_type=pred_type_formatted,
        MASE=mase,
        phase_shift=str(phase_shift)))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)


# Gets exactly the same records used to train the model in the pipeline
def get_train_records(series_id, horizon):
    target_name = series_id + '_' + str(horizon)
    connection, cursor = connect_to_postgres()
    cursor.execute(create_data_prep_table_sql.format(
        target_name=target_name,
        series_id=series_id,
        forecast_horizon=horizon))
    sql_get_data = '''

    SELECT features.*
      FROM data_preparation_{target_name} AS features
      JOIN record_ids
           ON features.observation_id::FLOAT = record_ids.observation_id::FLOAT
              AND record_ids.target = '{target_name}'
     WHERE record_ids.target = '{target_name}'
           AND model_id = {model_id}
           AND purpose = 'train'

    '''.format(target_name=target_name, model_id=1)
    df = get_df(sql_get_data)
    cursor.close()
    connection.close()
    return df


# Used to validate no error autocorrelation
def plot_train_errors(series_id, horizon):
    df = get_train_records(series_id, 1)

    target = series_id + '_' + str(horizon)

    # Get best model
    model_id, model_restored = restore_best_model(target)

    # Get ordered predictor names
    predictor_names = get_predictor_names(
        target=target,
        model_id=model_id)

    # Get and record prediction
    predictions = model_restored.predict(df[predictor_names])

    df['score'] = predictions
    df['error'] = df['score'] - df['target']

    colors = ['#DC0A16', '#FD8226', '#FDDA31', '#75A21D', '#42632B']
    sns.set_style(style='darkgrid')

    # Ensure graph and text is large enough to read
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    ax = plt.gca()

    plt.scatter(
        df['observation_id'],
        y=df['error'], c=colors[4],
        label='Error')

    ax.set_xlabel('Month(t)')
    ax.set_ylabel('Error')

    plt.title("{id} Cross Validation Model Train Errors".format(
        id=series_id.upper()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)


# ARIMA-like model predictions. Uses horizon-1 model recursively
def recursive_predictions(series_id, model_inputs):
    obs_id = model_inputs['observation_id'][0]
    target = series_id + "_1"

    # Get best one-ahead model
    model_id, model_restored = restore_best_model(target)

    # Get ordered predictor names
    predictor_names = get_predictor_names(
        target=target,
        model_id=model_id)

    lag_names = []
    for i in range(1, 13):
        lag_name = 'lag_' + str(i)
        lag_names.append(lag_name)

    month_names = []
    for i in range(0, 12):
        month_name = 'month_' + str(i)
        month_names.append(month_name)

    predictions = []
    lags = []
    for lag_name in lag_names:
        lag = model_inputs[lag_name][0]
        lags.append(lag)

    for horizon in range(0, 24):
        obs_id += 1

        month = obs_id % 12
        month_data = [0 for i in range(12)]
        month_data[month] = 1

        model_inputs_new = [obs_id] + lags[:12] + month_data
        prediction = model_restored.predict(np.array(model_inputs_new).reshape(1, -1))[0]
        predictions.append(prediction)
        lags = [prediction] + lags
    return predictions


# ARIMA-like model requires its own MASE calculation, since model logic is different
def get_recursive_mase(series_id, pred_type, phase_shift=False):
    model_inputs = get_model_inputs(
        series_id=series_id,
        forecast_horizon=0,
        pred_type=pred_type)

    if pred_type.lower() == 'validation':
        df = get_validation_data(series_id)
    else:
        df = get_oos_data(series_id, phase_shift=phase_shift)

    predictions = recursive_predictions(
        series_id=series_id,
        model_inputs=model_inputs)
    df['score'] = predictions
    df['error'] = abs(df['tourism'] - df['score'])
    avg_error = df['error'].mean()
    scale_factor = get_scale_factor(series_id)
    scaled_error = avg_error / scale_factor
    return scaled_error


# Trains horizon-1 model with time_index as only predictor
# TODO: Finish me
def train_trend_model(series_id):
    target_name = series_id + '_1_time_index'
    target_index = 0

    # Get training data
    connection, cursor = connect_to_postgres()
    cursor.execute(create_data_prep_table_sql.format(
        forecast_horizon=1,
        target_name=target_name,
        series_id=series_id))
    cursor.close()
    connection.close()
    df = get_df(pull_data_sql.format(target_name=target_name, series_id=series_id))

    # Run model training and evaluation pipeline
    print('Start training for {target} ... {target_index} of {total}'.format(
        target=target_name,
        target_index=target_index + 1,
        total=len(targets)))
    print('Target: {0}'.format(target_name))
    run_pipeline(df, ['observation_id'], target_name)
    print('Finished training {target}'.format(target=target_name))


# Show variation among various time series
def plot_all_ts_similarity(sample_type, series_type):
    assert (sample_type in ['in', 'oos'])
    assert (series_type in ['m', 'y', 'q'])
    sql_query = '''
        SELECT time_index,
               STDDEV_SAMP(tourism) AS tourism_stddev,
               AVG(tourism) AS tourism_avg
          FROM time_series
         WHERE LOWER(series_id) LIKE '%{series_type}%'
               AND LOWER(sample_type) LIKE '%{sample_type}%'
               AND time_index < 20
         GROUP BY time_index
         ORDER BY time_index ASC
    '''.format(
        sample_type=sample_type,
        series_type=series_type)
    df = get_df(sql_query)

    colors = ['#DC0A16', '#FD8226', '#FDDA31', '#75A21D', '#42632B']
    sns.set_style(style='darkgrid')

    # Ensure graph and text is large enough to read
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 12})
    ax = plt.gca()

    plot_1 = ax.plot(
        df['time_index'],
        df['tourism_avg'],
        label='Tourism',
        color=colors[0],
        linewidth=2)

    # Plot standard deviation above and below mean
    plt.fill_between(df['time_index'],
                     df['tourism_avg'] - df['tourism_stddev'],
                     df['tourism_avg'] + df['tourism_stddev'],
                     facecolor=colors[0],
                     alpha=0.25,
                     linewidth=0.0)

    ax.set_xlabel('Time')
    ax.set_ylabel('Tourism')

    plt.title("Tourism Similarity Over Time ({series_type}) From {sample_type} Sample".format(
        sample_type=sample_type.upper(),
        series_type=series_type.upper()))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)