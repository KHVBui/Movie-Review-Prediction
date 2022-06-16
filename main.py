import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

"""
Preparing the Data
"""
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

# Will test using 33% of the data
train, test = train_test_split(df_review_bal, test_size = 0.33,
                                random_state=42)
train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

# Need to convert movie review text into numerical vectors using 
# "bag of words" technique
# Stop word 'english' isn't comprehensive, can improve by tokenization
tfidf = TfidfVectorizer(stop_words = 'english')
train_x_vector = tfidf.fit_transform(train_x) # sparse matrix
# print(train_x_vector)
# print(pd.DataFrame.sparse.from_spmatrix(train_x_vector, index = train_x.index,
#                                         columns = tfidf.get_feature_names_out()))

# Don't fit to retain original mean and variance
test_x_vector = tfidf.transform(test_x)

"""
Model Selection: Classification Algorithms
"""
# Support Vector Machines (SVM)
# Input = text reviews as numerical vectors
# Output = sentiment
svc = SVC(kernel = 'linear')
svc.fit(train_x_vector, train_y)
# print(svc.predict(tfidf.transform(['A good movie']))) # positive
# print(svc.predict(tfidf.transform(['An excellent movie']))) # positive
# print(svc.predict(tfidf.transform(['I did not like this movie at all']))) # negative

# Decision Tree
# Input = text reviews as numerical vectors
# Output = sentiment
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_x_vector, train_y)

# Naive Bayes
# Input = text reviews as an array of numerical vectors
# Output = sentiment
gnb = GaussianNB()
gnb.fit(train_x_vector.toarray(), train_y)

# Logistic Regression
# Input = text reviews as numerical vectors
# Output = sentiment
log_reg = LogisticRegression()
log_reg.fit(train_x_vector, train_y)

# Evaluate the Mean accuracy of each model
# 'Test samples', 'True labels'
svc.score(test_x_vector, test_y) # 0.84
dec_tree.score(test_x_vector, test_y) # 0.65
gnb.score(test_x_vector.toarray(), test_y) # 0.63
log_reg.score(test_x_vector, test_y) # 0.83

# F1 Score = weighted avg of Precision and Recall 
# Used when False Negatives and False Positives are important
# Also helpful with data with imbalanced classes
f1_score(test_y, svc.predict(test_x_vector),
            labels = ['positive', 'negative'],
            average = None) # [0.84671533 0.83464567]

# Classification Report
classification_report(test_y,
                        svc.predict(test_x_vector),
                        labels = ['positive', 'negative'])

#Confusion Matrix
# [[TP FP] [FN TN]]
conf_mat = confusion_matrix(test_y,
                            svc.predict(test_x_vector),
                            labels = ['positive', 'negative'])
# [[290  45]
# [ 60 265]]

'''
Tuning the Model using GridSearchCV
'''
# Set the parameters
parameters = {'C': [1, 4, 8, 16, 32], 'kernel': ['linear', 'rbf']}
svc = SVC()
svc_grid = GridSearchCV(svc, parameters, cv = 5)

svc_grid.fit(train_x_vector, train_y)
# print(svc_grid.best_params_)
# print(svc_grid.best_estimator_)
# {'C': 1, 'kernel': 'linear'}
# SVC(C=1, kernel='linear')
# C = 1 is the default, so our SVM model was optimal already
