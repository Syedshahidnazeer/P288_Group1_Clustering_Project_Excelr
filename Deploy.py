from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Finalized_model.sav', 'rb'))

with open('Finalized_model.sav', 'wb') as f:
    pickle.dump(model, f)

def generate_navbar(active_page):
    pages = [
        {'name': 'Home', 'url': '/', 'active': active_page == 'home'},
        {'name': 'Clustering', 'url': '/clustering', 'active': active_page == 'clustering'},
        {'name': 'Classification', 'url': '/classification', 'active': active_page == 'classification'},
    ]
    return pages


@app.route('/')
def home():
    navbar = generate_navbar('home')
    return render_template('home.html', navbar=navbar)

@app.route('/clustering')
def clustering():
    navbar = generate_navbar('clustering')
    return render_template('clustering.html', navbar=navbar)

@app.route('/classification')
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
    
    navbar = generate_navbar('classification')
    return render_template('classification.html', navbar=navbar)


if __name__ == '__main__':
    app.run(debug=True)
