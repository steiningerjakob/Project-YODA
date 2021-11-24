from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/')
def index():
    return {'Yoda says:', 'You must unlearn what you have learned!'}



# PSEUDO CODE
# load the model.joblib file with basic model
# use pipeline to preprocess the data
# classification = model.predict(<picutre vectorized>)
# result = classifciation[0]
# return dict(prediction = pred)
