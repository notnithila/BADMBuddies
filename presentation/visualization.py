import preprocessing 
import seaborn as sns
import pandas as pd
#from IPython.display import display
import matplotlib.pyplot as plt

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

