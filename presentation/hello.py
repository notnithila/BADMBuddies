import pandas

df_outcomes_rankings = pandas.read_excel('/Users/sheetalsudhir/Documents/BADMBuddies/data/2022_County_Health_Rankings_Data.xlsx', "Outcomes & Factors Rankings")
#print(df_outcomes_rankings)

df_analytic_data = pandas.read_csv('/Users/sheetalsudhir/Documents/BADMBuddies/data/chr_analytic_data2022.csv')
print(df_analytic_data)

df_trends_data = pandas.read_csv('/Users/sheetalsudhir/Documents/BADMBuddies/data/chr_trends_csv_2022.csv')
print(df_trends_data)