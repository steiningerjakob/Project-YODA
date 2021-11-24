# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* Project-YODA/*.py

black:
	@black scripts/* Project-YODA/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr Project-YODA-*.dist-info
	@rm -fr Project-YODA.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      GOOGLE CLOUD STORAGE
# ----------------------------------
BUCKET_NAME=wagon-data-745-project-yoda

# TODO: to be verified if * works to upload everything in that directory
LOCAL_PATH=raw_data/

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
    @gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


# ----------------------------------
#      API
# ----------------------------------

run_api:
	uvicorn api.api:app --reload

# ----------------------------------
#  Create docker file and publsih via gcloud
# ----------------------------------

# set configurations for glcoud -> select the project ID required

config_gcloud:
	gcloud config set project project-yoda-333014

# build a docker image with M1/apple silicon chip
# select project ID and give an image name

docker_build_m1:
	docker build --platform linux/amd64 -t eu.gcr.io/project-yoda-333014/project_yoda .

# build a docker image with intel chip
# select project ID and give an image name

docker_build_intel:
	docker build --platform linux/amd64 -t eu.gcr.io/project-yoda-333014/project_yoda .

# run the docker image locally to double check that it's working, terminate the run before using docker_push
# select project ID and give an image name

docker_run:
	docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/project-yoda-333014/project_yoda

# push the docker image to gcloud
# select project ID and give an image name

docker_push:
	docker push eu.gcr.io/project-yoda-333014/project_yoda

# deploy the image on gcloud run
# select project ID and give an image name

docker_deploy:
	gcloud run deploy --image eu.gcr.io/project-yoda-333014/project_yoda --platform managed --region europe-west1
