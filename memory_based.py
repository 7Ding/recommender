import pandas as pd

# 1. Preprocessing
# read csv file
# ISO-8859-1 : Europe characters incoding
df = pd.read_csv('ProcessedData2.csv', encoding='ISO-8859-1')

# Delete data, if data in preprocessed data "Rating" has value '0'
df = df[df['Rating'] > 0]

# Normalization Rating 1~10 to 1~5
df['Rating'] = df['Rating'].apply(lambda x: x/2 if x > 5 else x) 

# Character->number mapping for efficient calculation and reduced memory usage
df['user_id'] = df['User-ID'].astype('category').cat.codes
df['book_id'] = df['ISBN'].astype('category').cat.codes

print(df.head())

from sklearn.metrics.pairwise import cosine_similarity

# User - book item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='book_id', values='Rating')

# Fill book items that the user has not rated with 0
user_item_matrix.fillna(0, inplace=True)

# Cosine Similarity and matrix between items
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Recommendations based on books rated by a user. Recommending 10 books
def book_recommend (user_id, recommend_num):
    try:
        # Load raitng of data (specific user)
        user_ratings = user_item_matrix.loc[user_id]
        similar_items = item_similarity_df.dot(user_ratings).sort_values(ascending=False)

        recommendation = similar_items.index[~similar_items.isin(user_ratings[user_ratings > 0].index)]
        recommend_indices = recommendation[:recommend_num]

        book_recommend = df[df['book_id'].isin(recommend_indices)][['Title', 'Author', 'Year', 'Publisher']]

        return book_recommend.reset_index(drop=True)

    except KeyError:
        print(f"User ID {user_id} is not exist in dataset")
        return pd.DataFrame()

   
# Type user_id and type the number of books
def get_user_id():
    try:
        input_id = int(input("Please type the user ID (num) : "))
        input_recommend_num = int(input("Please type number of recommendation book : "))
        
        output = book_recommend(input_id, input_recommend_num)

        if not output.empty:
            print("Recommended books list :")
            print(output)
        else:
            print("There is no recommendation")

    except ValueError:
        print("False input value")
    
# Test result
get_user_id()
