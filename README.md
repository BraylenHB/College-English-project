# College-English-project
Website for Vervint
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data: Client and employee profiles
clients = pd.DataFrame({
    'client_id': [1, 2],
    'desired_characteristics': ["patient, experienced, good listener", 
                                "creative, innovative, tech-savvy"]
})

employees = pd.DataFrame({
    'employee_id': [101, 102, 103],
    'attributes': ["experienced, good listener, empathetic",
                   "creative, tech-savvy, innovative",
                   "patient, empathetic, adaptable"]
})

# Interaction history: Satisfaction scores (1-5 scale)
interactions = pd.DataFrame({
    'client_id': [1, 2, 1],
    'employee_id': [101, 102, 103],
    'satisfaction_score': [4, 5, 3]
})

# Function to calculate text similarity using TF-IDF
def calculate_similarity(client_text, employee_texts):
    vectorizer = TfidfVectorizer()
    combined = [client_text] + employee_texts
    tfidf_matrix = vectorizer.fit_transform(combined)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarities

# Function to match a client with the best employee
def match_client(client_id, weight_similarity=0.7, weight_history=0.3):
    # Get the client's desired characteristics
    client_row = clients[clients['client_id'] == client_id]
    client_desc = client_row['desired_characteristics'].values[0]
    
    # Calculate similarity scores
    similarity_scores = calculate_similarity(client_desc, employees['attributes'].values)
    employees['similarity_score'] = similarity_scores
    
    # Add prior interaction scores for the client
    interaction_scores = interactions[interactions['client_id'] == client_id]
    employees['interaction_score'] = employees['employee_id'].map(
        interaction_scores.set_index('employee_id')['satisfaction_score']
    ).fillna(0)
    
    # Normalize scores (scale to 0-1)
    employees['similarity_score'] /= employees['similarity_score'].max()
    employees['interaction_score'] /= 5  # Max satisfaction score is 5
    
    # Calculate final score with weights
    employees['final_score'] = (
        weight_similarity * employees['similarity_score'] +
        weight_history * employees['interaction_score']
    )
    
    # Sort employees by final score
    best_match = employees.sort_values('final_score', ascending=False).iloc[0]
    return best_match[['employee_id', 'final_score']]

# Example: Match client 1
best_match = match_client(client_id=1)
print("Best match for Client 1:")
print(best_match)

# Example: Match client 2
best_match = match_client(client_id=2)
print("\nBest match for Client 2:")
print(best_match)
