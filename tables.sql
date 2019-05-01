  BEGIN;

 CREATE TABLE IF NOT EXISTS time_series(
        series_id varchar(4),
        sample_type varchar(20),
        time_index int,
        tourism float,
PRIMARY KEY(series_id, sample_type, time_index));

 COMMIT;
  BEGIN;

 CREATE TABLE IF NOT EXISTS model_storage(
        target varchar(100),
        model_id int,
        model_binary bytea,
PRIMARY KEY(target, model_id));

 COMMIT;
  BEGIN;

 CREATE TABLE IF NOT EXISTS predictors(
        target varchar(100),
        model_id int,
        predictor_index int,
        predictor varchar(100),
PRIMARY KEY(target, model_id, predictor_index, predictor));

 COMMIT;
  BEGIN;

 CREATE TABLE IF NOT EXISTS record_ids(
        target varchar(100),
        model_id int,
        purpose varchar(100),
        observation_id varchar(100),
PRIMARY KEY(target, model_id, observation_id));

 COMMIT;
  BEGIN;

 CREATE TABLE IF NOT EXISTS fit_performance(
        target varchar(100),
        model_id int,
        purpose varchar(100),
        loss_function varchar(100),
        performance float,
PRIMARY KEY(target, model_id, purpose, loss_function));

 COMMIT;
  BEGIN;

 CREATE TABLE IF NOT EXISTS models(
        target varchar(100),
        model_id int,
        create_date date,
PRIMARY KEY(target, model_id, create_date));

 COMMIT;
  BEGIN;

 CREATE TABLE max_time
     AS
 SELECT series_id,
        MAX(time_index) AS max_time
   FROM time_series
  WHERE sample_type = 'in'
  GROUP BY series_id;

 COMMIT;

 CREATE TEMP TABLE unified_ts
     AS
 SELECT time_series.series_id,
        time_series.sample_type,
        time_series.time_index,
        CASE
        WHEN time_series.sample_type = 'in'
        THEN time_series.time_index
        ELSE
        max_time.max_time + time_series.time_index + 1
        END AS new_time_index,
        time_series.tourism
   FROM time_series
   JOIN max_time
        ON time_series.series_id = max_time.series_id;

 CREATE TEMP TABLE lags
     AS
 SELECT series_id,
        sample_type,
        new_time_index,
        tourism,
        LAG(tourism, 12) OVER lifetime AS lag_12,
        LAG(tourism, 4) OVER lifetime AS lag_4,
        LAG(tourism, 1) OVER lifetime AS lag_1
   FROM unified_ts
 WINDOW lifetime AS (PARTITION BY series_id
        ORDER BY new_time_index ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
  ORDER BY sample_type ASC, time_index ASC;

  BEGIN;

 CREATE TABLE scale_factor
     AS
 SELECT series_id,
        SUM(CASE
        WHEN LOWER(series_id) LIKE '%m%'
        THEN ABS(tourism - lag_12)
        WHEN LOWER(series_id) LIKE '%q%'
        THEN ABS(tourism - lag_4)
        WHEN LOWER(series_id) LIKE '%y%'
        THEN ABS(tourism - lag_1)
        ELSE NULL END) /
        SUM(CASE
        WHEN LOWER(series_id) LIKE '%m%'
             AND lag_12 IS NOT NULL
        THEN 1
        WHEN LOWER(series_id) LIKE '%q%'
             AND lag_4 IS NOT NULL
        THEN 1
        WHEN LOWER(series_id) LIKE '%y%'
             AND lag_1 IS NOT NULL
        THEN 1
        ELSE NULL END) AS error
   FROM lags
  GROUP BY series_id;

 COMMIT;