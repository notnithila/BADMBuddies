import preprocessing
from presentation.preprocessing import get_final_df 
import seaborn as sns
import pandas as pd
import preprocessing
import impute
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = preprocessing.get_final_df(preprocessing.df_outcomes_rankings, preprocessing.df_outcomes_subrankings, preprocessing.df_ranked_measure, preprocessing.df_additional_measure)
#display(df)
#df = preprocessing.get_final_df(preprocessing.df_outcomes_rankings, preprocessing.df_outcomes_subrankings, preprocessing.df_ranked_measure, preprocessing.df_additional_measure)
#display(df)

fig, ax = plt.subplots(figsize=(84,84))
corr_mat = df.corr()
sns.heatmap(corr_mat)

#final_df.corr()

sorted_mat = corr_mat.unstack().sort_values()
print(sorted_mat)


final_df = preprocessing.get_final_df(preprocessing.df_outcomes_rankings, preprocessing.df_outcomes_subrankings, preprocessing.df_ranked_measure, preprocessing.df_additional_measure)
final_df = impute(final_df)

# get rid of outliers


df_no_outliers = final_df[(np.abs(stats.zscore(final_df)) < 3).all(axis=1)]

# input data into model for feature selection

# create feature variables
X = df_no_outliers.drop('Preventable hospital stays Preventable Hospitalization Rate',axis= 1)
y = df_no_outliers['Preventable hospital stays Preventable Hospitalization Rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)
inital_predictions = model.predict(X_test)

print('mean_squared_error : ', mean_squared_error(y_test, inital_predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, inital_predictions))

# import numpy as np 
# fig, ax = plt.subplots(figsize=(84,84))
# mask2 = np.triu(np.ones_like(corr_mat))
# sns.heatmap(corr_mat2, annot=True, vmax=1, vmin=-1, center=0, mask=mask2)

# tower thingy with outlier data 
corr_mat2 = df_no_outliers.corr(method = 'spearman')

plt.figure(figsize=(50,50))
new_heatmap2 = sns.heatmap(corr_mat2[['Preventable hospital stays Preventable Hospitalization Rate']].sort_values(by='Preventable hospital stays Preventable Hospitalization Rate', ascending=False,), vmin=-1, vmax=1, annot = True, cmap='BrBG')
new_heatmap2.set_title("Test", fontdict={'fontsize':18}, pad=16)

#split dataframe into medical and social variables

