import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from my_custom_module import Combine_Transformer, Numeric_Transformer, Categorical_Transformer_Freq, Categorical_Transformer_other, Text_Transformer


df = pd.read_csv("amazon_supplement.csv")

### import data, fill na
#Ensure Consistent Data Types
df.replace(to_replace=[None, 'nan', 'None', ''], value=np.nan, inplace=True)
df.dropna(subset=['rating'], inplace=True)

# 'review' number < 'rating' number, can use prediction to fillna, same 'rating', the null review number equals to the average
rating_record = df['rating'].loc[df['review_num'].isnull()].unique()
rating_review_table = {}

for i in rating_record:
    rating_review_table[i] = int(df['review_num'].loc[df['rating']==i].mean())
    
rating_fill_idx = df.loc[df['review_num'].isnull()].index.tolist()
for idx in rating_fill_idx:
    #print(df['review_num'][idx])
    df['review_num'][idx] = rating_review_table[df['rating'][idx]]
    #print(df['review_num'][idx])


### create popularity score
rating = df['rating']

# Apply log transformation to review_num
log_review_num = np.log1p(df['review_num'])

# Normalize rating and log_review_num
# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit and transform the data
log_review_num_norm = pd.Series(scaler.fit_transform(log_review_num.values.reshape(-1, 1)).flatten())
rating_num_norm = pd.Series(scaler.fit_transform(rating.values.reshape(-1, 1)).flatten())

# Create a composite popularity score
# Here, we give equal weight to rating and review number, but you can adjust the weights based on your preference
popularity_score = 0.5 * log_review_num_norm + 0.5 * rating_num_norm

# drop the data wo 'Popularity_Score'
series = pd.Series(popularity_score, name='Popularity_Score')
df['Popularity_Score'] = series
df.dropna(subset=['Popularity_Score'], inplace=True)


### create feature and target
features = df.columns.tolist()
target = 'Popularity_Score'
features.remove(target)
for i in ['title', 'rating', 'review_num', 'link']:
    features.remove(i)

# find the threshold to define 'popular'
# Convert to numpy array for easy manipulation
data = np.array(df[target])
# Sort the data in descending order
sorted_data = np.sort(data)[::-1]
# Calculate the index for the top 20%
top_20_index = int(len(sorted_data) * 0.2)
# Find the threshold value
threshold = sorted_data[top_20_index - 1]  # -1 because index is zero-based

# function to judge if it is popular or not
def Popular_judge(x):
        if type(x) == float: 
            if x >= threshold:
                return 1
            else:
                return 0
df['Popularity_Score_class'] = df.Popularity_Score.map(Popular_judge)
target = 'Popularity_Score_class'

X = df[features]
y = df[target]

steps = [('tsf', Combine_Transformer()),
         ('rsc', MinMaxScaler()),
         ('gdbc', xgb.XGBClassifier(scale_pos_weight=4,        # ratio is the imbalance ratio
                                    n_estimators=100,         # Number of trees
                                    learning_rate=0.01,        # Learning rate (shrinkage)
                                    max_depth=5,               # Maximum depth of each tree
                                    min_child_weight=5,        # Minimum sum of instance weight (hessian) needed in a child
                                    gamma=0.1,                 # Minimum loss reduction required to make a further partition on a leaf node of the tree
                                    subsample=0.8,             # Subsample ratio of the training instance
                                    colsample_bytree=0.8,      # Subsample ratio of columns when constructing each tree
                                    reg_alpha=0.1,             # L1 regularization term on weights
                                    reg_lambda=1.0,            # L2 regularization term on weights
                                    objective='binary:logistic',# Specify the learning task and the corresponding learning objective
                                    eval_metric='logloss',     # Evaluation metric
                                    random_state=42))]  
model = Pipeline(steps)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
