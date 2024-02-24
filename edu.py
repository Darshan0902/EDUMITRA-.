from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
import pandas as pd

# Load data
df = pd.read_csv("EDUMITRA _USERS_DATA.csv")  # Replace "your_dataset.csv" with the actual dataset file path

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

# Define the list of skills from the database
skills_list = df['Skills'].unique().tolist()

# Create a dictionary to store users indexed by their skills
users_by_skill = defaultdict(list)
for _, row in df.iterrows():
    users_by_skill[row['Skills']].append(row['id'])

# Prepare the data for the decision tree classifier
X = df['Skills'].values.reshape(-1, 1)  # Feature: Skills
y = df['id']  # Target: User ID

# One-hot encode the skills
encoder = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories gracefully
X_encoded = encoder.fit_transform(X)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Prompt the user to input their desired skill
selected_skill = input("Enter a skill to find recommended users: ")

# Get recommended users for the selected skill
recommended_user_ids = recommend_users_for_skills([selected_skill])
recommended_users_info = df[df['id'].isin(recommended_user_ids)][['id', 'first_name', 'email', 'gender']]

# Print recommended users
if not recommended_users_info.empty:
    print(recommended_users_info.to_string(index=False))
else:
    print("No users found with the selected skill.")
