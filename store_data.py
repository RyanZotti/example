import argparse
import functools
import os
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

from functions import get_df, connect_to_postgres


ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True,help="Postgres host")
args = vars(ap.parse_args())
host = args['host']

files = ['monthly_in.csv',
         'monthly_oos.csv',
         'quarterly_in.csv',
         'quarterly_oos.csv',
         'yearly_in.csv',
         'yearly_oos.csv']

# Assumes files unzipped to repo directory
project_path = os.getcwd()

# Save each file to Postgres
connection_str = 'postgresql://postgres:postgres@{host}:5432/predata_tourism'.format(host=host)
postgres_connection = create_engine(connection_str)
for file in files:
    # Read the data from file
    file_path = functools.reduce(
        os.path.join,
        [project_path, 'tourism_data', '{file}'.format(file=file)])
    df = pd.read_csv(filepath_or_buffer=file_path)

    # Write the data to the Postgres database
    table = file.split('.')[0]
    df.to_sql(name='{table}'.format(table=table),
              con=postgres_connection,
              index=False)

# Save the data in an RDBMS-friendly way
# List of tables to iterate through
tables = ['monthly_in',
         'monthly_oos',
         'quarterly_in',
         'quarterly_oos',
         'yearly_in',
         'yearly_oos']

# Create connection to Postgres
connection, cursor = connect_to_postgres(host=host)

# Reuseable SQL insert statement
sql_insert = """
     BEGIN;
    INSERT INTO time_series(
           series_id,
           sample_type,
           time_index,
           tourism)
    VALUES ('{series_id}',
           '{sample_type}',
           {time_index},
           {tourism});
    COMMIT;"""

for table in tables:
    series_type, sample_type = table.split('_')
    df = get_df('SELECT * FROM {table_name}'.format(
            table_name=table),
            host=host)
    series_ids = list(df.columns.values)
    for time_index, row in df.iterrows():
        for series_id in series_ids:
            tourism = row[series_id]
            if not np.isnan(tourism):
                cursor.execute(sql_insert.format(
                    series_id=series_id,
                    sample_type=sample_type,
                    time_index=time_index,
                    tourism=tourism))

cursor.close()
connection.close()
print('Finished successfully')