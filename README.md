The task is all about to Design a simple ML model to recommend mentors (CLAT toppers) to aspirants based on user profiles.

## Problem Statement

CLAT aspirants often struggle to find mentors who align with their academic goals, preparation level, and learning preferences. This project aims to build a basic Machine Learning model that recommends the top 3 most suitable mentors to each aspirant using similarity-based matching.


## Features

- Personalized mentor recommendations based on aspirant profiles
- Profile features include:
  - Preferred subjects
  - Target law colleges
  - Preparation level
  - Learning style
- Uses **Cosine Similarity** to rank mentors
- Visualizes the top 3 mentor matches in a bar chart


## Tech Stack

- Python 
- Pandas (Data Handling)
- Scikit-learn (Preprocessing & Similarity)
- Matplotlib (Visualization)

## Proces to be followed 
Our first dtep is to collect the data for our model development; make the data for the aspirants with the partcular data fields.
---
next step is to provide the feature processing , to come up with the relevancy of the data that are required for the system , so that
will able to connect with the requiremnts for our analysis part.
----
There are various segments as well in order to make "Data Cleaning" in order to get the meaningful data as well.
----
Next will perform "Cosine Similarity" in order to rank the mentors.
Cosine Similarity is a metric used to measure how similar two vectors are, regardless of their size or magnitude. It calculates the cosine of the angle between the two vectors in an inner product space.
We use cosine similarity to compare the aspirant's profile vector with each mentor’s vector. The mentors with the highest similarity scores are considered the best match for recommendation.
---
Performing the analysis part to recommend Top 3 Mentors
---
After getting this result, the next step is to visualize the analysis with the help of Graphical Representation
for that purpose , we uses "matplotlib" and "seaborn" libraries.

## Conclusion :
This project presents a smart and personalized way to help law aspirants find the right mentors based on their unique learning needs and goals. 
By using basic machine learning and cosine similarity, the system matches aspirants with mentors who share similar interests, target colleges, and learning styles. 
This not only saves time but also makes mentorship more effective and engaging.

As the system grows with more users and feedback, it can become even smarter, offering better recommendations and creating a strong support network for CLAT and other law entrance exam aspirants.
It’s a step towards making preparation more focused, guided, and successful.

## Code :
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Aspirant Profile (Input)
aspirant = {
    'preferred_subjects': ['Constitutional Law', 'Legal Reasoning'],
    'target_colleges': ['NLSIU', 'NLU Delhi'],
    'prep_level': 'Intermediate',
    'learning_style': 'Visual'
}

# Mock Mentors Dataset
data = [
    {'mentor': 'Mentor A', 'preferred_subjects': ['Constitutional Law'], 'target_colleges': ['NLSIU'], 'prep_level': 'Advanced', 'learning_style': 'Visual'},
    {'mentor': 'Mentor B', 'preferred_subjects': ['Legal Reasoning', 'English'], 'target_colleges': ['NLU Delhi'], 'prep_level': 'Intermediate', 'learning_style': 'Auditory'},
    {'mentor': 'Mentor C', 'preferred_subjects': ['Logical Reasoning'], 'target_colleges': ['NLU Jodhpur'], 'prep_level': 'Beginner', 'learning_style': 'Kinesthetic'},
    {'mentor': 'Mentor D', 'preferred_subjects': ['Constitutional Law', 'Legal Reasoning'], 'target_colleges': ['NLSIU', 'NLU Delhi'], 'prep_level': 'Intermediate', 'learning_style': 'Visual'},
]

mentors_df = pd.DataFrame(data)
all_data = mentors_df.copy()
all_data.loc[-1] = aspirant
all_data.index = all_data.index + 1
all_data = all_data.sort_index()

mlb = MultiLabelBinarizer()
subjects_encoded = mlb.fit_transform(all_data['preferred_subjects'])
colleges_encoded = mlb.fit_transform(all_data['target_colleges'])prep_level_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
learning_style_map = {'Visual': 0, 'Auditory': 1, 'Kinesthetic': 2}
prep_encoded = all_data['prep_level'].map(prep_level_map)
learn_encoded = all_data['learning_style'].map(learning_style_map)

import numpy as np
features = np.hstack((subjects_encoded, colleges_encoded, 
                      prep_encoded.values.reshape(-1, 1), 
                      learn_encoded.values.reshape(-1, 1)))

aspirant_vector = features[0].reshape(1, -1)
mentor_vectors = features[1:]

similarities = cosine_similarity(aspirant_vector, mentor_vectors)[0]
mentors_df['similarity'] = similarities


top_mentors = mentors_df.sort_values(by='similarity', ascending=False).head(3)
print("Top 3 Mentor Recommendations:")
print(top_mentors[['mentor', 'similarity']])

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of the plot
sns.set(style="whitegrid")

# Create a bar plot of top 3 mentors by similarity score
plt.figure(figsize=(10, 6))
sns.barplot(x='similarity', y='mentor', data=top_mentors, palette='viridis')

# Set plot labels and title
plt.title('Top 3 Mentor Recommendations Based on Similarity', fontsize=16)
plt.xlabel('Similarity Score', fontsize=14)
plt.ylabel('Mentor', fontsize=14)

# Display the plot
plt.show()



## Summary of the approach being used 
The project begins by collecting essential information from the law aspirant, including their preferred subjects, target colleges, current preparation level, and learning style. 
This profile helps in understanding the aspirant’s needs and preferences for effective mentorship matching.

Next, a mock dataset of mentors (such as past CLAT toppers) is created. Each mentor's profile includes the same fields as the aspirant, ensuring a consistent basis for comparison. 
To prepare the data for machine learning, all categorical information is encoded into numerical format. Multi-label fields like subjects and colleges are transformed using MultiLabelBinarizer, 
while single-choice fields like preparation level and learning style are mapped to integer values.

After encoding, all features are combined into vectors for both the aspirant and the mentors. The system then calculates the cosine similarity between the aspirant’s feature vector and each mentor’s vector. 
Cosine similarity measures how closely aligned the two profiles are, regardless of the overall scale of their data. Higher similarity scores indicate better matches.

Finally, the mentors are ranked based on their similarity scores, and the top 3 mentors with the highest scores are recommended to the aspirant.
As a bonus, a bar chart visualization is included to help users visually understand the match quality. This approach is simple, interpretable, and effective for providing personalized mentorship recommendations.

