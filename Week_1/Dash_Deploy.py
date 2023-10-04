from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Week_1\Finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/presentation')
def presentation():
    return render_template('presentation.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    features = [float(x) for x in request.form.values()]
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Make prediction using the loaded model
    prediction = model.predict(df)

    # Convert prediction to list for JSON response
    prediction = prediction.tolist()

    return render_template('index.html', prediction_text='The input belongs to cluster {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
