from typing import final
import pandas as pd
#from IPython.display import display
from impute import impute 

# read excel file sheets into separate df
excel_path = '/Users/avaneeshs/Documents/BADMBuddies/data/2022_County_Health_Rankings_Data.xlsx'
df_outcomes_rankings = pd.read_excel(excel_path, "Outcomes & Factors Rankings")
df_outcomes_subrankings = pd.read_excel(excel_path, "Outcomes & Factors SubRankings")
df_ranked_measure = pd.read_excel(excel_path, "Ranked Measure Data")
df_additional_measure = pd.read_excel(excel_path, "Additional Measure Data")

# read csv data into pandas df
df_analytic_data = pd.read_csv('/Users/avaneeshs/Documents/BADMBuddies/data/chr_analytic_data2022.csv')
df_trends_data = pd.read_csv('/Users/avaneeshs/Documents/BADMBuddies/data/chr_trends_csv_2022.csv')


def rename_columns(columns, first_row):
    prefix = ""
    for c in columns:
        if "Unnamed:" not in c:
            prefix = c
        cIndex = columns.index(c)
        first_row[cIndex] = prefix + " " + first_row[cIndex]
    return 

def apply_rename_columns(df):
    x = df.columns.tolist()
    y = list(df.loc[0])
    rename_columns(x, y)
    
    df.columns = y

    df = df.iloc[1: , :]
    df_ranked_measure.reset_index(drop = True)
    
    return df

def get_final_df(df_outcomes_rankings, df_outcomes_subrankings, df_ranked_measure, df_additional_measure):
    df_outcomes_rankings = apply_rename_columns(df_outcomes_rankings)
    df_outcomes_subrankings = apply_rename_columns(df_outcomes_subrankings)
    df_ranked_measure = apply_rename_columns(df_ranked_measure)
    df_additional_measure = apply_rename_columns(df_additional_measure)

    massive_df = df_outcomes_rankings.merge(df_outcomes_subrankings, on=[" FIPS"," County"," State"])
    massive_df = massive_df.merge(df_ranked_measure, on=[" FIPS"," County"," State"])
    massive_df = massive_df.merge(df_additional_measure, on=[" FIPS"," County"," State"])

    adjusted_massive_df = impute(massive_df)

    #display(adjusted_massive_df)
    return adjusted_massive_df

final_df = get_final_df(df_outcomes_rankings, df_outcomes_subrankings, df_ranked_measure, df_additional_measure)
final_df = impute(final_df)

# get rid of outliers
import numpy as np
from scipy import stats

temp = final_df[(np.abs(stats.zscore(final_df)) < 3).all(axis=1)]

# input data into model for feature selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

# create feature variables
X = temp.drop('Preventable hospital stays Preventable Hospitalization Rate',axis= 1)
y = temp['Preventable hospital stays Preventable Hospitalization Rate']

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
corr_mat2 = temp.corr(method = 'spearman')

plt.figure(figsize=(50,50))
new_heatmap2 = sns.heatmap(corr_mat2[['Preventable hospital stays Preventable Hospitalization Rate']].sort_values(by='Preventable hospital stays Preventable Hospitalization Rate', ascending=False,), vmin=-1, vmax=1, annot = True, cmap='BrBG')
new_heatmap2.set_title("Test", fontdict={'fontsize':18}, pad=16)