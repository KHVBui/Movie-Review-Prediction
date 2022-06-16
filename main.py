import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


df_review = pd.read_csv('IMDB Dataset.csv')
# print(df_review)

# Extract 10,000 rows to process at a time
df_positive = df_review[df_review['sentiment'] == 'positive'][:9000]
df_negative = df_review[df_review['sentiment'] == 'negative'][:1000]

# Combine and reset the indices of the pos and neg rows
df_review_imb = pd.concat([df_positive, df_negative], ignore_index = True)
# print(df_review_imb)

# Under sample since more positive than negative rows
# fit_resample needs a 2D dataframe for the x-value
rus = RandomUnderSampler(random_state = 0)
df_review_bal, df_review_bal['sentiment'] = rus.fit_resample(
                                                df_review_imb[['review']],
                                                df_review_imb['sentiment'])
# print(df_review_imb.value_counts('sentiment'))
# print(df_review_bal.value_counts('sentiment'))

# Test 33% of the data
train, test = train_test_split(df_review_bal, test_size = 0.33,
                                random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# Need to convert movie review text into numerical vectors using 
# "bag of words" technique
# Stop word 'english' isn't comprehensive, and may need to be customized
tfidf = TfidfVectorizer(stop_words = 'english')
train_x_vector = tfidf.fit_transform(train_x)
# print(train_x_vector)
print(pd.DataFrame.sparse.from_spmatrix(train_x_vector, index = train_x.index,
                                    columns = tfidf.get_feature_names_out()))






