import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions 
from sklearn import tree

st.write("""
# Insurance Prediction App
Look at how the Model works. Pick some features from the Menu to the left to find out if you are 
eligible for low monthly insurance premiums?
""")

st.sidebar.header('User Input Parameters')
name=st.sidebar.text_input('Please Enter Your Name')

def user_input_features():
    Insurance_Type = st.sidebar.selectbox('Type of Insurance', ['Basic', 'Extended', 'Premium'])
    Car_Type = st.sidebar.selectbox('Vehicle Type', ['SUV', 'Four-Door Car','Sports Car', 'Two-Door Car', 'Luxury SUV', 'Luxury Car'])
    Driving = st.sidebar.selectbox('Location of Most Driving', ['Rural', 'Urban', 'Suburban'])
    Reason = st.sidebar.selectbox('Claim Reason', ['Collision', 'Hail', 'Scratch/Dent', 'Other', 'None'])
    Total_Claim_Amount = st.sidebar.slider('Total Claim Amount', 0, 10000, 100)
    #Claim_Amount = st.sidebar.slider('Claim Amount', 0, 10000, 100)
    data = {'Insurance_Type': Insurance_Type,
            'Car_Type': Car_Type,
            'Driving': Driving,
            'Reason': Reason,
            'Total Claim Amount': Total_Claim_Amount}
            #'Claim Amount': Claim_Amount}
    features = pd.DataFrame(data, index=[0])
    return features

#display_inputs = features
#st.subheader('User Input parameters')
#st.write(display_inputs)
inputs = user_input_features()

# Split it on the median because the mean will distribute the positive and negative class more evenly.
inputs['Total Claim Amount'] = np.where(inputs['Total Claim Amount'] > 383.95 , 1, 0)
#inputs['Claim Amount'] = np.where(inputs['Claim Amount'] > 578.02 , 1, 0)

subcategory_user = inputs.loc[:,inputs.dtypes == object]

# Instantiate the OneHotEncoder
ohe_user = OneHotEncoder()
# Fit the OneHotEncoder to non numeric columns.
encoded_user = ohe_user.fit_transform(subcategory_user)
# Convert from sparse matrix to dense
dense_array_user = encoded_user.toarray()

# Let us breakup the above array into a single list
cats_user = []
for i in ohe_user.categories_:
    #print(i)
    for j in i:
        #print(j)
        cats_user.append(j)

# Put into a dataframe to get column names
encoded_df_inputs = pd.DataFrame(dense_array_user, columns=cats_user, dtype=int)

# Reset the index so the values start from 0
inputs.reset_index(drop = True, inplace = True)

# Add to the main dataframe
inputs_OHE = pd.concat([inputs,encoded_df_inputs], axis = 1)

#Make a list of the non numeric columns
tcolumns_user = []
for i in inputs.columns:
    if inputs[i].dtype == object:
        tcolumns_user.append(i)
inputs_OHE.drop(tcolumns_user, axis = 1 , inplace = True)

kbest15_user = ['Basic',
 'SUV',
 'Total Claim Amount',
 'Four-Door Car',
 'Claim Amount',
 'Extended',
 'Premium',
 'Sports Car',
 'Two-Door Car',
 'Luxury SUV',
 'Luxury Car',
 'Collision',
 'Hail',
 'Other',
 'Suburban']

user_df = pd.DataFrame(np.zeros((1,15)), columns = kbest15_user)

for i in inputs_OHE.columns:
    #print(i)
    if i in kbest15_user:
        #print(i)
        user_df[i] = np.where(user_df[i] == 0, 1, 0)

#st.subheader('User Input parameters')
#st.write(inputs)

df = pd.read_csv('Auto_Insurance_Claims_Sample(Capstone).csv')
# Loop through the above columns and perform the operation that will get us a good distribution of data
for i in df.columns: 
    if ((df[i].dtype == int) | (df[i].dtype == float)):
        # Bin the column based on the median
        df[i] = np.where(df[i] > df[i].median() , 1, 0)
# Drop the columns mentioned above.

df.drop(['Customer','Country','State Code','Response','Policy Type'], axis = 1, inplace = True)        

X = df.drop('Monthly Premium Auto', axis = 1)
y = df['Monthly Premium Auto']

# We are going to create a Train Remainder Validation Test Split. 
from sklearn.model_selection import train_test_split
X_remain, X_test, y_remain, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1, stratify = y)

X_train, X_validation, y_train, y_validation = train_test_split(X_remain,y_remain, test_size = 0.3, random_state = 1, stratify = y_remain)

subcategory = X_train.loc[:,X_train.dtypes == object]

from sklearn.preprocessing import OneHotEncoder

# Instantiate the OneHotEncoder
ohe = OneHotEncoder()

# Fit the OneHotEncoder to non numeric columns.
encoded = ohe.fit_transform(subcategory)
# Convert from sparse matrix to dense
dense_array = encoded.toarray()

# Let us breakup the above array into a single list
cats = []
for i in ohe.categories_:
    #print(i)
    for j in i:
        #print(j)
        cats.append(j)

# Put into a dataframe to get column names
encoded_df = pd.DataFrame(dense_array, columns=cats, dtype=int)

# Reset the index so the values start from 0
X_train.reset_index(drop = True, inplace = True)

# Add to the main dataframe
X_train_OHE = pd.concat([X_train,encoded_df], axis = 1)


#Make a list of the non numeric columns
tcolumns = []
for i in df.columns:
    if df[i].dtype == object:
        tcolumns.append(i)

#Drop the non numeric columns but keep the newly created OHE columns
X_train_OHE.drop(tcolumns, axis = 1 , inplace = True)

subcategory_valid = X_validation.loc[:,X_validation.dtypes == object]

# Fit the OneHotEncoder to non numeric columns.
encoded_valid = ohe.fit_transform(subcategory_valid)
# Convert from sparse matrix to dense
dense_array_valid = encoded_valid.toarray()
#print(dense_array_valid)
# Put into a dataframe to get column names
encoded_df_valid = pd.DataFrame(dense_array_valid, columns=cats, dtype=int)
#encoded_df_valid.head()

X_validation.reset_index(drop = True, inplace = True)
# add to the main dataframe
X_validation_OHE = pd.concat([X_validation,encoded_df_valid], axis = 1)

#Drop the non numerical columns.
X_validation_OHE.drop(tcolumns, axis = 1 , inplace = True)

subcategory_remain = pd.DataFrame(X_remain[tcolumns])

# Fit the OneHotEncoder to non numeric columns.
encoded_remain = ohe.fit_transform(subcategory_remain)
# Convert from sparse matrix to dense
dense_array_remain = encoded_remain.toarray()
# Put into a dataframe to get column names
encoded_df_remain = pd.DataFrame(dense_array_remain, columns=cats, dtype=int)

X_remain.reset_index(drop = True, inplace = True)
# add to the main dataframe
X_remain_OHE = pd.concat([X_remain,encoded_df_remain], axis = 1)

#Drop the non numerical columns.
X_remain_OHE.drop(tcolumns, axis = 1 , inplace = True)

subcategory_test = pd.DataFrame(X_test[tcolumns])

# Fit the OneHotEncoder to non numeric columns.
encoded_test = ohe.fit_transform(subcategory_test)
# Convert from sparse matrix to dense
dense_array_test = encoded_test.toarray()
# Put into a dataframe to get column names
encoded_df_test = pd.DataFrame(dense_array_test, columns=cats, dtype=int)

X_test.reset_index(drop = True, inplace = True)
# add to the main dataframe
X_test_OHE = pd.concat([X_test,encoded_df_test], axis = 1)

#Drop the non numerical columns.
X_test_OHE.drop(tcolumns, axis = 1 , inplace = True)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions

# Instatiate
selector = SelectKBest(score_func = f_classif, k = 'all')
# Fit
fit = selector.fit(X_train_OHE,y_train)
# 3 decimal places
set_printoptions(precision = 3)
print(fit.scores_)
#transform into the shape we want
features = fit.transform(X_train_OHE)

print(features[0:,:])

# First, create a table of the feature names and their regression coefficients
kbest_table = pd.DataFrame(list(X_train_OHE.columns)) # Copy out the feature names from the data
# transpose is required for the shapes to match when inserting the data.
kbest_table.insert(len(kbest_table.columns),"KBest",fit.scores_.transpose()) # insert the logreg coefficients

kbest15 = list(kbest_table.sort_values(by = 'KBest', ascending = False).head(15)[0])
X_kbest_train_15 = X_train_OHE.loc[:,kbest15]
X_kbest_validation_15 = X_validation_OHE.loc[:,kbest15]
X_kbest_remain_15 = X_remain_OHE.loc[:,kbest15]
X_kbest_test_15 = X_test_OHE.loc[:,kbest15]

insurance_logit_final = LogisticRegression(C= 0.1, penalty = 'l1', solver = 'liblinear',random_state = 42)
insurance_logit_final.fit(X_kbest_remain_15,y_remain)

insurance_DT_final = DecisionTreeClassifier(max_depth = 6)
insurance_DT_final.fit(X_kbest_remain_15,y_remain)

prediction_logit = insurance_logit_final.predict(user_df)
prediction_proba_logit = insurance_logit_final.predict_proba(user_df)
logit_pred= pd.DataFrame(prediction_logit,columns = ['Class'] )
logit_prob= pd.DataFrame(prediction_proba_logit,columns = ['Low Premium','High Premium'] )

prediction_DT = insurance_DT_final.predict(user_df)
prediction_proba_DT = insurance_DT_final.predict_proba(user_df)

if prediction_logit == 1:
    st.header(f' Thank you {name} for applying for coverage at DA Insurance Group. Please reach out to one of our representativates for confirmation of your approval.')
    st.header('Please call : 909-909-0909 for more information about your approval')
else:
    st.title('Congratulations!!!')
    st.header(f'Congratulations {name}!! You have been automatically approved for low price auto coverage at DA Insurance Group')
    st.header('Please fill in the rest of the application to get your final quote')
    email=st.text_input('Please Enter Your Email')
    phone = st.text_input('Please Enter Your Number')
    Address = st.text_input('Please Enter Your Address')

data_classes = { 'Premium Amount': ['Low Premium','High Premium']
}
classes = pd.DataFrame(data_classes)

classl=st.sidebar.checkbox('Show Class Labels')
if classl == True:
    st.subheader('Class Labels')
    st.write(classes)

prediction=st.sidebar.checkbox('Show Predictions')
if prediction == True:
    st.subheader('Prediction')
    st.write(logit_pred)
    #st.write(prediction_DT)

proba=st.sidebar.checkbox('Show Probability')
if proba == True:
    st.subheader('Prediction Probability')
    st.write(logit_prob)
    #st.write(prediction_proba_DT)

treel = st.sidebar.checkbox('Show Decision Tree')
if treel == True:
    dt_plot = tree.export_graphviz(insurance_DT_final, out_file = None,
            feature_names=X_kbest_train_15.columns ,
            rounded=True,
            impurity=False,
            filled=True,
            label = 'all',
            proportion = False,
            node_ids = True)
    st.graphviz_chart(dt_plot)
    