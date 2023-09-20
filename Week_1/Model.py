# Importing the required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset into a pandas dataframe
data = pd.read_csv('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Dataset\IQR.csv')

# Creating a DataFrame with the loaded data
df = pd.DataFrame(data)

data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [1, 2, 3, 4, 5],
    'target': ['class1', 'class2', 'class1', 'class2', 'class1']
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Load the dataset into a pandas dataframe
x = pd.read_csv('E:\P288_Clustering_Project\P288_Group1_Clustering_Project_Excelr\Dataset\IQR.csv')

# Define the encoding dimension
encoding_dim = 32

# Define the input layer
input_img = Input(shape=(100,))

# Define the encoded layer
encoded = Dense(encoding_dim, activation='relu')(input_img)

# Define the decoded layer
decoded = Dense(64, activation='sigmoid')(encoded)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Create the encoder model
encoder = Model(input_img, encoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Fit the autoencoder model
autoencoder.fit(x, x,
                epochs=50,
                batch_size=256,
                shuffle=True)

# Use the encoder to reduce the dimensionality of the data
x_reduced = encoder.predict(x)

# Apply K-means clustering on the reduced data
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_reduced)