# Project YODA

<img width="997" alt="Screenshot 2021-12-09 at 15 43 14" src="https://user-images.githubusercontent.com/77202477/145417537-d92db467-3282-48a9-9994-c03093e21c4d.png">

## Description

Project YODA is an AI engine for image classification and generation based on LEGO minifigures. 

## The team

This application was created for educational purposes by [Yiping](https://github.com/ypzhangescp), [Daniel](https://github.com/Daniel-Lars), [Hicham](https://github.com/HicZer) and myself. 

## Usage

**Image classification:** Simply go to [our website](https://updated-frontend-zl47dkr23a-ew.a.run.app/), select an image for prediction, hit *Classify me!*, and enjoy Yoda's predictions! Currently, the application supports around 40 different characters from the Star Wars, Marvel, Harry Potter and Jurrasic Park universes. Go to the Kaggle dataset linked below for a full list of supported characters.

**Image generation:** Check out this short presentation for initial results of the [image generation engine](https://github.com/steiningerjakob/Project-YODA/blob/master/notebooks/Project%20Yoda%20-%20Demo%20Day%20Presentation_vFINAL.pdf).

## Technologies used

- Backend:
  - Convolutional neural network models using Tensorflow Keras
  - Google Colab and Compute Engine for GPU-powered model training  
  - Prediction API using FastAPI
  - Data and model storage on Google Cloud Storage  
- Frontend in [separate GitHub repo](https://github.com/Daniel-Lars/Project-YODA_frontend):
  - Streamlit 
- Deployment:
  - Docker images deployed on Google Cloud Run

### The dataset

[Kaggle LEGO Minifigures](https://www.kaggle.com/ihelon/lego-minifigures-classification)
