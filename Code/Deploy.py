from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    messages = {
        0: "The input belongs to cluster 0, This cluster has countries with relatively low CO2 emissions, GDP, and energy usage. The average days to start a business is around 24. The health expenditure as a percentage of GDP is around 0.064 and the average health expenditure per capita is 619. The infant mortality rate is relatively low at 0.012 and the lending interest is 0.103. The life expectancy for females and males is 78 and 73 years respectively.",
        1: "The input belongs to cluster 1, This cluster represents countries with higher CO2 emissions, GDP, and energy usage compared to Cluster 0. The average days to start a business is around 18. The health expenditure as a percentage of GDP is higher at 0.087 and the average health expenditure per capita is significantly higher at 1182. The infant mortality rate is lower at 0.005 and the lending interest is lower at 0.079. The life expectancy for females and males is significantly higher at 82 and 77 years respectively.",
        2: "The input belongs to cluster 2, This cluster represents countries with even lower CO2 emissions, GDP, and energy usage compared to Cluster 0. The average days to start a business is around 34. The health expenditure as a percentage of GDP is slightly lower at 0.058 and the average health expenditure per capita is the lowest at 63. The infant mortality rate is higher at 0.072 and the lending interest is higher at 0.157. The life expectancy for females and males is significantly lower than other clusters at 56 and 54 years respectively.",
        3: "The input belongs to cluster 3, This cluster represents countries with high CO2 emissions, GDP, and energy usage but lower than Cluster 1. The average days to start a business is around 30. The health expenditure as a percentage of GDP is lower at 0.051 and the average health expenditure per capita is relatively low at 237. The infant mortality rate is relatively low at 0.026 and the lending interest is slightly higher at 0.125 compared to Cluster 1 but lower than Cluster 2. The life expectancy for females and males is in between other clusters at around 73 and 67 years respectively.",
        4: "The input belongs to cluster 4, This cluster represents countries with low CO2 emissions, GDP, and energy usage similar to Cluster 0 but slightly higher on average. The average days to start a business is around 31 which is higher than Cluster 0 but lower than others. The health expenditure as a percentage of GDP is slightly lower than Cluster 0 at around 0.060 and the average health expenditure per capita is relatively low at around 182 which is higher than Cluster 2 but lower than others. The infant mortality rate is relatively low at around 0.025 which is similar to Cluster 3 but lower than others, and the lending interest rate is slightly higher than others except for Cluster2 at around .140 . The life expectancy for females and males are similar to Cluster3"
    }

    # Get the message for the predicted cluster
    prediction_text = messages[prediction[0]]

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/visualizations')
def visualizations():
    df = pd.read_csv('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Dataset\IQR.csv')
    df.copy=df
    numerical_cols = df.columns[df.dtypes != 'object'].values
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numerical_cols])
    kmeans = KMeans(n_clusters=5, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    df['Cluster'] = clusters
    labels_kmeans=kmeans.labels_

    sse = []
    list_k = list(range(1, 10))

    for k in list_k:
        km = KMeans(n_clusters=k)
        km.fit(df_scaled)
        sse.append(km.inertia_)

    fig1 = go.Figure(data=go.Scatter(x=list_k, y=sse, mode='lines+markers'))

    # Generate your plots
    fig2 = px.scatter(df, x=df.index, y='Cluster', color='Cluster')
    
    # Create a DataFrame for cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_values(ascending=False).reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    fig3 = px.bar(cluster_counts, x='Cluster', y='Count')

    fig4 = px.imshow(df.corr(), title='Heatmap of feature correlations')

    fig5 = go.Figure(data=go.Scatter3d(
    x=df.iloc[:, 0],  # assuming the DataFrame has at least three columns
    y=df.iloc[:, 1],
    z=df.iloc[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        color=labels_kmeans,  # set color to the K-Means cluster labels
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
))                
    # Convert the figures to HTML div strings
    plot1_div = pio.to_html(fig1, full_html=False)
    plot2_div = pio.to_html(fig2, full_html=False)
    plot3_div = pio.to_html(fig3, full_html=False)
    plot4_div = pio.to_html(fig4, full_html=False)
    plot5_div = pio.to_html(fig5, full_html=False)
    # Pass the div strings into your template
    return render_template('visualizations.html', plot1_div=plot1_div, plot2_div=plot2_div,plot3_div=plot3_div, plot4_div=plot4_div,plot5_div=plot5_div)

@app.route('/project')
def project():
        return render_template('project.html')
if __name__ == '__main__':
    app.run(port=5000, debug=True)
