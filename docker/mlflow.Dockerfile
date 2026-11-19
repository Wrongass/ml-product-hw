FROM ghcr.io/mlflow/mlflow:v3.16.0

# postgres driver and s3 client are not in the official image
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.44.7
