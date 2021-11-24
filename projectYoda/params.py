### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[SW] [Stockholm] [steiningerjakob] prod_v0.1"

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'

root_dir = "../raw_data/"

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-745-project-yoda'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location - not yet implemented
BUCKET_DATA_PATH = 'raw_data'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'classification'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
