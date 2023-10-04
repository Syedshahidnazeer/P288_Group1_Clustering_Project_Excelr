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

@app.route('/random_input')
def random_input():
    return render_template('random_input.html')

@app.route('/generate_random_input', methods=['POST'])
def generate_random_input():

    for column, stats in stats_dict.items():
        # Define dictionaries for the remaining columns
    stats_dict = {
    'energy_usage': {
        'mean': 18268.08358,
        'std': 14774.93527,
        'min': 8.000,
        'max': 46913.625
    },
    'GDP': {
        'mean': 5.968253e+10,
        'std': 8.148197e+10,
        'min': 6.310127e+07,
        'max': 2.213750e+11
    },
    'health_exp_percent_GDP': {
        'mean': 0.06303,
        'std': 0.02214,
        'min': 0.00800,
        'max': 0.11800
    },
    'health_exp_percapita': {
        'mean': 393.259430,
        'std': 446.539198,
        'min': 2.000,
        'max': 1291.125
    },
    'infant_mortality_rate': {
        'mean': 0.031404,
        'std': 0.028215,
        'min': 0.002000,
        'max': 0.107500
    },
    'lending_interest': {
        'mean': 0.125866,
        'std': 0.048682,
        'min': 0.020500,
        'max': 0.226500
    },
    'life_expectancy_female': {
        'mean': 71.363536,
        'std': 10.438511,
        'min': 44.000,
        'max': 87.000
    },
    'life_expectancy_male': {
        'mean': 66.599112,
        'std': 9.141010,
        'min': 43.000,
        'max': 88.000
    },
    'mobile_phone_usage': {
        'mean': 0.572319,
        'std': 0.456606,
        'min': 0.000000,
        'max': 1.950000
    },
    'population_15-64': {
        'mean': 0.624969,
        'std': 0.066771,
        'min': 0.474000,
        'max': 0.835000
    },
    'population_65+': {
        'mean': 0.071684,
        'std': 0.047735,
        'min': 0.003000,
        'max': 0.211500
    },
    'population_total': {
        'mean': 13571050,
        'std': 16640770,
        'min': 18876,
        'max': 49034730
    },
    'tourism_inbound': {
        'mean': 1825707000,
        'std': 2285833000,
        'min': 700000,
        'max': 6341000000
    },
    'tourism_outbound': {
        'mean': 1191360000,
        'std': 1488580000,
        'min': 200000,
        'max': 4058250000
    },
    'birth_rate': {
        'mean': 0.022595,
        'std': 0.011100,
        'min': 0.007000,
        'max': 0.053000
    },
    'internet_usage': {
        'mean': 0.230806,
        'std': 0.253983,
        'min': 0.000000,
        'max': 1.000000
    },
    'population_0-14': {
        'mean': 0.302762,
        'std': 0.102152,
        'min': 0.118000,
        'max': 0.500000
    },
    'population_urban': {
        'mean': 0.562924,
        'std': 0.244712,
        'min': 0.082000,
        'max': 1.000000
    }
}

    random_input_values = {}

    for column, stats in stats_dict.items():
        # Extract statistics for the current column
        mean = stats['mean']
        std = stats['std']
        min_value = stats['min']
        max_value = stats['max']

        # Generate a random value following a normal distribution
        random_value = np.random.normal(loc=mean, scale=std)

        # Ensure the generated value is within the min-max range
        random_value = max(min(random_value, max_value), min_value)

        # Add the random value to the dictionary
        random_input_values[column] = random_value

    print("Generating random input values:")
    print(random_input_values)

    random_df = pd.DataFrame([random_input_values])

    # Make prediction using the loaded model
    random_prediction = model.predict(random_df)

    # Convert prediction to list for JSON response
    random_prediction = random_prediction.tolist()

    return render_template('random_input.html', random_input_values=random_input_values, prediction_text=f'The random input belongs to cluster {random_prediction[0]}')


    return render_template('random_input.html', random_input_values=random_input_values)

    return render_template('index.html', prediction_text='The input belongs to cluster {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(port=5000, debug=True)
