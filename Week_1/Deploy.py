from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Week_1\Finalized_model.sav', 'rb'))

with open('Finalized_model.sav', 'wb') as f:
    pickle.dump(model, f)

@app.route('/')
def home():
    return render_template('index.html')

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

# Define the route to render the random input page
@app.route('/random_input')
def random_input():
    return render_template('random_input.html')

# Define the route to generate random input
@app.route('/generate_random_input', methods=['POST'])
def generate_random_input():
    # Generate random input values
    random_input_values = {
        'birth_rate': np.random.uniform(10, 50),
        'business_tax_rate': np.random.uniform(0, 30),
    }

    return render_template('index.html', prediction_text='Random Input Values: {}'.format(random_input_values))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
