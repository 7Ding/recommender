import pandas as pd

# 1. Preprocessing
# read csv file
# ISO-8859-1 : Europe characters incoding
df = pd.read_csv('ProcessedData2.csv', encoding='ISO-8859-1')
print(df.head())

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
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Training dataset : User - book item matrix
user_item_matrix = train_df.pivot_table(index='user_id', columns='book_id', values='Rating').fillna(0)
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix.T)



# Rocommendation function to user
def book_recommend_knn(user_id, recommend_num):
    if user_id not in user_item_matrix.index:
        print("{user_id} is not exist in dataset")
        return pd.DataFrame
    
    user_ratings = user_item_matrix.loc[user_id]
    rated_books = user_ratings[user_ratings>0].index.tolist()

    recommended_books_indices = []
    recommended_books_distances = []

    for book_id in rated_books:
        distances, indices = knn.kneighbors(user_item_matrix.T.iloc[book_id].values.reshape(1, -1), n_neighbors=recommend_num+1)
        similar_books_indices = indices.flatten()[1:]  # except 0 (item itself)
        similar_books_distances = distances.flatten()[1:]  # 

        recommended_books_indices.extend(similar_books_indices)
        recommended_books_distances.extend(similar_books_distances)

    # List of similar books (deleted same item) : number of items : recommend_num
    recommended_books_indices = list(set(recommended_books_indices))[:recommend_num]
    recommended_books_info = df[df['book_id'].isin(recommended_books_indices)][['Title', 'Author', 'Year', 'Publisher']]
    return recommended_books_info.reset_index(drop=True)


def recommend_test():
    input_recommend_num = int(input("Please type number of recommendation book : "))
    for i in range(3):
        test_user_id = test_df['user_id'].sample(1).iloc[0]
        print("Recommendation test", i+1, test_user_id)
        output = book_recommend_knn(user_id=test_user_id, recommend_num=input_recommend_num)
        print(output)
        print("\n")

recommend_test()

""" 
# memory_based2.py에서 삭제
# Cosine Similarity and matrix between items
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Recommendation function to user
def book_recommend(user_id, num_recommendations=15):
        
    try: 
        # Loading the information of user rating (user_id) / similar_items : Find similar items
        user_ratings = user_item_matrix.loc[user_id]  
        similar_items = item_similarity_df.dot(user_ratings).sort_values(ascending=False)  
        
        # If there is any book that user_id user already rated, except that book
        recommendations = similar_items.index[~similar_items.index.isin(user_ratings[user_ratings > 0].index)]  
        # Top n Recommendations   n: changeable
        recommended_indices = recommendations[:num_recommendations]  
    
        # Information to display in the recommendation list : Book Title, Author, Year, Publisher
        book_recommend = df[df['book_id'].isin(recommended_indices)][['Title', 'Author', 'Year', 'Publisher']]
    
        return book_recommend.reset_index(drop=True)
    
    except:
        print(f"User {user_id} is not exist in dataset")
        return pd.DataFrame()
"""

"""
# Checking whether modeling was successful
# Random user
random_user_id = train_df['user_id'].sample(1).iloc[0]
recommended_books_output = book_recommend(user_id=random_user_id, num_recommendations=15)
print("\nRecommendation books list : ")
print(recommended_books_output)

# modeling test using test dataset 
def recommendation_test(test_user_id):
    if test_user_id not in user_item_matrix.index:
        print(f"Input for testing {test_user_id} is not exist.")
        return
    
    # Recommendation 
    recommended_books_output = book_recommend(user_id=test_user_id, num_recommendations=15)
    
    print(f"\nRecommended books for {test_user_id}:")
    print(recommended_books_output)

# Recommendation for random test user
def repeat_test(test_num):
    for i in range(test_num):
        random_test_user = test_df['user_id'].sample(1).iloc[0]

        print("Test {1+1} : book list for {random_test_user}")
        recommendation_test(random_test_user)

input_num = int(input("Please type the number to loop: "))
repeat_test(input_num)
"""