import preprocessing 
import seaborn as sns
import pandas as pd
from IPython.display import display
#import matplotlib.pyplot as plt

df = preprocessing.get_final_df(preprocessing.df_outcomes_rankings, preprocessing.df_outcomes_subrankings, preprocessing.df_ranked_measure, preprocessing.df_additional_measure)
display(df)
