from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv("EDUMITRA _USERS_DATA.csv")

# Prepare the data for the decision tree classifier
X = df['Skills'].values.reshape(-1, 1)  # Feature: Skills
y = df['id']  # Target: User ID

# One-hot encode the skills
encoder = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories gracefully
X_encoded = encoder.fit_transform(X)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Create a dictionary to store users indexed by their skills
users_by_skill = defaultdict(list)
for _, row in df.iterrows():
    users_by_skill[row['Skills']].append(row['id'])

# Function to recommend users based on the input skills
def recommend_users_for_skills(skills):
    # Encode the input skills
    skills_encoded = encoder.transform([[skill] for skill in skills])

    # Predict users with similar skills using the trained classifier
    predicted_users = clf.predict(skills_encoded)

    # Get all users with common skills
    recommended_user_ids = set()
    for user_id in predicted_users:
        recommended_user_ids.update(users_by_skill[df.loc[df['id'] == user_id, 'Skills'].values[0]])
    return recommended_user_ids

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_users', methods=['POST'])
def recommend_users():
    data = request.get_json()
    selected_skill = data['selected_skill']

    # Get recommended users for the selected skill
    recommended_user_ids = recommend_users_for_skills([selected_skill])
    recommended_users_info = df[df['id'].isin(recommended_user_ids)][['id', 'first_name', 'email', 'gender']]

    # Prepare response
    if not recommended_users_info.empty:
        response = recommended_users_info.to_dict(orient='records')
    else:
        response = {"message": "No users found with the selected skill."}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
