import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define emotion labels
labels = [
    'female_calm', 'male_calm', 'female_happy', 'male_happy',
    'female_sad', 'male_sad', 'female_angry', 'male_angry',
    'female_fearful', 'male_fearful'
]

# Initialize and fit the LabelEncoder
lb = LabelEncoder()
lb.fit(labels)

# Save the classes to a .npy file
np.save('label_encoder_classes.npy', lb.classes_)

print("Label encoder classes saved successfully.")
