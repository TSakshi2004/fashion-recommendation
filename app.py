from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("fashion_data_1000.csv")

# Prepare the feature matrix for similarity
features = df[['Category', 'Color', 'Style']]
encoder = OneHotEncoder()
encoded = encoder.fit_transform(features).toarray()
similarity = cosine_similarity(encoded)

# Recommendation function
def get_recommendations(item_name):
    if item_name not in df['Name'].values:
        return []
    index = df[df['Name'] == item_name].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # top 5 excluding self
    recommendations = df.iloc[[i[0] for i in sorted_scores]][['Name', 'Category', 'Color', 'Style', 'Price']]
    return recommendations.to_dict(orient='records')

@app.route("/", methods=["GET", "POST"])
def home():
    selected_item = None
    recommendations = []
    
    if request.method == "POST":
        selected_item = request.form.get("item_name")
        recommendations = get_recommendations(selected_item)
        
    items = df['Name'].unique()
    return render_template("index.html", items=items, selected_item=selected_item, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
