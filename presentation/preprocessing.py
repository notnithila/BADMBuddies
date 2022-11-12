import pandas as pd

# read excel file sheets into separate df
excel_path = '/Users/sheetalsudhir/Documents/BADMBuddies/data/2022_County_Health_Rankings_Data.xlsx'
df_outcomes_rankings = pd.read_excel(excel_path, "Outcomes & Factors Rankings")
df_outcomes_subrankings = pd.read_excel(excel_path, "Outcomes & Factors SubRankings")
df_ranked_measure = pd.read_excel(excel_path, "Ranked Measure Data")
df_additional_measure = pd.read_excel(excel_path, "Additional Measure Data")

# read csv data into pandas df
df_analytic_data = pd.read_csv('/Users/sheetalsudhir/Documents/BADMBuddies/data/chr_analytic_data2022.csv')
df_trends_data = pd.read_csv('/Users/sheetalsudhir/Documents/BADMBuddies/data/chr_trends_csv_2022.csv')