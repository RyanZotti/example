predictors = [
    'observation_id',
    'lag_1',
    'lag_2',
    'lag_3',
    'lag_4',
    'lag_5',
    'lag_6',
    'lag_7',
    'lag_8',
    'lag_9',
    'lag_10',
    'lag_11',
    'lag_12',
    'month_0',
    'month_1',
    'month_2',
    'month_3',
    'month_4',
    'month_5',
    'month_6',
    'month_7',
    'month_8',
    'month_9',
    'month_10',
    'month_11'
    ]

create_data_prep_table_sql = '''

  BEGIN;

   DROP TABLE IF EXISTS data_preparation_{target_name};

 COMMIT;
  BEGIN;

 CREATE TABLE data_preparation_{target_name}
     AS
(SELECT time_index AS observation_id,
        tourism AS target,
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

 COMMIT;
'''

pull_data_sql='''
 SELECT observation_id,
        target AS {target_name},
        lag_1,
        lag_2,
        lag_3,
        lag_4,
        lag_5,
        lag_6,
        lag_7,
        lag_8,
        lag_9,
        lag_10,
        lag_11,
        lag_12,
        month_0,
        month_1,
        month_2,
        month_3,
        month_4,
        month_5,
        month_6,
        month_7,
        month_8,
        month_9,
        month_10,
        month_11
   FROM data_preparation_{target_name}
  WHERE lag_12 IS NOT NULL
        AND future_record_count > 24
'''