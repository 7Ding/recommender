import pandas as pd

# 1. Preprocessing
# read csv file
# ISO-8859-1 : Europe characters incoding
df = pd.read_csv('ProcessedData2.csv', encoding='ISO-8859-1')

# Delete data, if data in preprocessed data "Rating" has value '0'
# Normalization Rating 1~10 to 1~5
df = df[df['Rating'] > 0]
df['Rating'] = df['Rating'].apply(lambda x: x/2 if x > 5 else x) 

# Character->number mapping for efficient calculation and reduced memory usage
df['user_id'] = df['User-ID'].astype('category').cat.codes
df['book_id'] = df['ISBN'].astype('category').cat.codes

print(df.head())


from sklearn.model_selection import train_test_split

# Separated into (80% learning, 20% testing)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Checking training and testing dataset
print("Train Data:")
print(train_df.head())
print("\nTest Data:")
print(test_df.head())

# Checking normalization
import matplotlib.pyplot as plt

print(df[['user_id', 'book_id', 'Rating']])
plt.figure(figsize=(8,6))
plt.hist(df['Rating'], bins=5, edgecolor='black')
plt.title("Normalizated Rating 1~5")
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

from sklearn.metrics.pairwise import cosine_similarity

# Training dataset : User - book item matrix
user_item_matrix = train_df.pivot_table(index='user_id', columns='book_id', values='Rating').fillna(0)

# Cosine Similarity and matrix between items
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Recommendation function to user
def recommend_books(user_id, num_recommendations=15):
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} is not exist in dataset")
        return pd.DataFrame()

    # Loading the information of user rating (user_id) / similar_items : Find similar items
    user_ratings = user_item_matrix.loc[user_id]  
    similar_items = item_similarity_df.dot(user_ratings).sort_values(ascending=False)  
    # If there is any book that user_id user already rated, except that book
    recommendations = similar_items.index[~similar_items.index.isin(user_ratings[user_ratings > 0].index)]  
    # Top n Recommendations   n: changeable
    recommended_indices = recommendations[:num_recommendations]  
    
    # Information to display in the recommendation list : Book Title, Author, Year, Publisher
    recommended_books = df[df['book_id'].isin(recommended_indices)][['Title', 'Author', 'Year', 'Publisher']]
    
    return recommended_books.reset_index(drop=True)

# Checking whether modeling was successful
# Random user
random_user_id = train_df['user_id'].sample(1).iloc[0]
recommended_books_output = recommend_books(user_id=random_user_id, num_recommendations=15)
print("\nRecommendation books list : ")
print(recommended_books_output)

# modeling test using test dataset 
def test_recommendation(test_user_id):
    if test_user_id not in user_item_matrix.index:
        print(f"Input for testing {test_user_id} is not exist.")
        return
    
    # Recommendation 
    recommended_books_output = recommend_books(user_id=test_user_id, num_recommendations=15)
    
    print(f"\nRecommended books for {test_user_id}:")
    print(recommended_books_output)

# Recommendation for random test user
random_test_user = test_df['user_id'].sample(1).iloc[0]
test_recommendation(test_user_id=random_test_user)