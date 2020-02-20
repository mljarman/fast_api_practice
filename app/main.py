from fastapi import FastAPI
from joblib import load
import logging
import pandas as pd
import uvicorn

app = FastAPI()

# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.DEBUG)

# Load pickled models
nn = load('./baseline_compressed.pkl')
tfidf = load('./tfidf.pkl')

@app.post("/recommend/{input}")
async def recommend(input: str):
    try:
        series_input = pd.Series(input)
        vect_input = tfidf.transform(series_input)
        recommended_strains = nn.kneighbors(vect_input.todense())[1][0].tolist()
        string_suggestion = " ".join(str(i) for i in recommended_strains)
        return {"needs":input, "strains":string_suggestion}
    except:
        my_logger.error("Something went wrong!")
        return {"prediction": "error"}
