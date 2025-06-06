import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load dataset
df = pd.read_csv("C:\Users\LENOVO\OneDrive\Desktop\AML CA II\fashion_data_1000.csv")

# Prepare features
features = df[['Category', 'Color', 'Style']]

# Create and fit encoder
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(features).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(encoded_features)

# Save encoder and similarity matrix to .pkl files
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("Saved encoder.pkl and similarity.pkl!")
