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

