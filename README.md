## File Overview

This project comprises the following items: 

* `Data Exploration and Visualization.ipynb` contains visual and qualitative analysis. If you're in a hurry and just want the results, check out this file first.
* `functions.py` contains boilerpate and project-agnostic code for training models. I use TPOT to automate model training and selection, and I use Postgres to save and restore model binaries. I also have several light integration tests in this file that can be extended to model deployment. 
* `exploration_functions.py` contains graphs related to visualizing the results of the models. Normally I would put these in the Jupyter notebook, but I wanted to keep that clean and wanted fewer code distractions. 
* `data_config.py` contains re-usable data preparation code. This code is re-used in training and later in visualization of predictions.
* `store_data.py` takes data from files and stores them in Postgres.
* `tables.sql` contains the Postgres create-table statements to initialize the Postgres database. 
* `train_models.py` train all of a time series id's forecast-horizon models using this script. This version of the script runs locally.
* `spark_train_models.py` same as `train_models.py` except that it runs on Apache Spark. The code is very similar. 
* `emr_configs.sh` Installs Anaconda and other essential libraries on AWS's EMR / Spark cluster. 


## Data Preparation

First I normalize the data and store it in Postgres in a table called `time_series`. I then apply some light feature engineering to create the following predictors:

* time index / observation id
* month category (modulo operation on time index)
* lags 1 through 12

## Machine Learning

I used two approaches to fitting models. 

**Approach 1.** ARIMA-style predict-one-period-ahead model that recursively forecasts 1 to 24 months out. This approach is more intuitive, easier to code, and less taxing from a data-resource perspective. 

**Approach 2.** A separate model for each series id and for each forecast horizon. This approach is very complicated, and results in thousands of models. Unfortunately, this approach has 10-15% better performance than the ARIMA-style approach, so I kept it in my code base. 

I did not use ARIMA models or any of the classical time series types of models. I've used ARIMA models in the past (NBA sports betting, March Madness, and stock market forecasting, to namew a few) and I have never seen good predictive results compared to other, more modern techniques. So I didn't bother here, though I would have pursued if I had more time or if Predata had a preference for the classical models.

I use TPOT to automate the ML process. 

## Data Infrastructure

I have code that can run locally or on a one-off AWS / EMR Apache Spark cluster. Spark is my preferred tool at work, but I wasn't able to finish the Spark infrastructure for this project because I ran out of time. This isn't a paid assessment, so I set a 2-week time limit. 

Nonetheless, Spark would provide a much needed speed-up for my second machine learning approach. Also, Spark on AWS is cheap -- I use 30 m4.large EC2 instances on the AWS Spot Market. With 75% cost savings off the on-demand price, my 30-node Spot Market Spark cluster costs me less than a dollar an hour, and I can turn it on and off at will and re-size as needed.

## Instructions

### Install and Set Up Local Database

```
# Install a specific version of Postgres on OSX
brew install postgresql@9.5

# Set enviornment variables
HOST="localhost"
DB="predata_tourism"

# Create the database
psql -U postgres -h $HOST -c "create database ${DB};"

# Build the tables 
psql -h $HOST -U postgres -d ${DB} -a -f tables.sql
```

### Load All Data to Local Postgres DB

```
python store_data.py --host localhost
```

### Run Model Pipeline Locally

```
python train_models.py
```

### Set Up Remote Postgres Server (optional)

```
# Create an Ubuntu EC2 instance and install Postgres on it
sudo apt-get update
sudo apt-get install -y build-essential libsm6 libxext6
sudo apt-get install -y git awscli postgresql postgresql-contrib libpq-dev
sudo apt-get install -y libfontconfig1 libxrender1 awscli

# Start Postgres server
sudo service postgresql start

# Set enviornment variables
HOST="localhost"
DB="predata_tourism"

# Create the database and set basic password
sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'postgres';"
psql -U postgres -h $HOST -c "CREATE DATABASE ${DB};"

# Allow connections (note the ' ' around the * are critical)
echo "listen_addresses = '*'" | sudo tee --append /etc/postgresql/9.5/main/postgresql.conf > /dev/null
echo "host all all 0.0.0.0/0 md5" | sudo tee --append /etc/postgresql/9.5/main/pg_hba.conf > /dev/null

# Restart Postgres for changes to take effect
sudo service postgresql restart

# Tell Postres on EC2 it's ok for things to conncet to it
echo "host all all 0.0.0.0/0 md5" >> /etc/postgresql/9.5/main/pg_hba.conf
sudo service postgresql restart

# From another server (like your laptop)
# Dowloand Predata-provided Kaggle tourism data from S3
LOCAL_FOLDER=/Users/ryanzotti/Documents/repos/predata-tourism/tourism_data
S3_FOLDER='s3://ryanzotti-data-science/predata-tourism/tourism_data'
aws s3 sync ${S3_FOLDER} ${LOCAL_FOLDER} 

# Build the tables
HOST="34.234.97.29"
DB="predata_tourism"
psql -h $HOST -U postgres -d ${DB} -a -f tables.sql

# This step can take a few minutes because the data 
# is normalized and has to transfer over the network
python store_data.py --host $HOST
```

### Spark, AWS, and EMR (optional but substantially faster)

```
# Switch to hadoop (not hdfs) user once logged into master node
# The hadoop user will have Anaconda and TPOT installed
sudo su hadoop
cd /home/hadoop

# Download repo
git clone https://github.com/RyanZotti/predata-tourism
cd predata-tourism

# Add spark variables to Anaconda install
export SPARK_HOME=/usr/lib/spark
export REPO_DIR='/home/hadoop/predata-tourism'
export PYTHONPATH=$SPARK_HOME/python/:$SPARK_HOME/python/build:$PYTHONPATH:${REPO_DIR}/

# Tell master to tell workers the python version they should use
export PYSPARK_PYTHON=~/anaconda/bin/python
export PYSPARK_DRIVER_PYTHON=~/anaconda/bin/python

spark-submit --py-files /home/hadoop/predata-tourism/functions.py,/home/hadoop/predata-tourism/data_config.py spark_train_models.py

spark-submit --py-files /home/hadoop/predata-tourism/functions.py,/home/hadoop/predata-tourism/data_config.py \
    --total-executor-cores 30 \
    spark_train_models.py 

```

## Tips

* Print Linux distro: `cat /proc/version`
* Ubuntu 16.04 AMI: `ami-66506c1c`
* View EMR bootstrap error logs: `vi /emr/instance-controller/log/bootstrap-actions/1/stderr`
* Get yarn logs: `yarn logs -applicationId application_1522025572332_0008 > application_1522025572332_0008.log`

## References

* Interpeting MASE: [link](https://stats.stackexchange.com/questions/124365/interpretation-of-mean-absolute-scaled-error-mase)
* Python ARIMA: [link](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
* Explains differences between Statsmodels, Pandas, and Matplotlib's autocorrelation visualizations: [link](https://stackoverflow.com/questions/27541290/bug-of-autocorrelation-plot-in-matplotlib-s-plt-acorr)
* Install Anaconda on EMR using setup script in S3 [link](https://dziganto.github.io/zeppelin/spark/zeppelinhub/emr/anaconda/tensorflow/shiro/s3/theano/bootstrap%20script/EMR-From-Scratch/) 
* Install Anaconda on EMR using boostramp script in S3 and boto3 [link](https://medium.com/@datitran/quickstart-pyspark-with-anaconda-on-aws-660252b88c9a) 
* Transfer Postgres DB from one host to another [link](https://stackoverflow.com/questions/1237725/copying-postgresql-database-to-another-server)

