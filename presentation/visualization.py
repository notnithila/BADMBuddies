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
medical_cols = ["Premature death Years of Potential Life Lost Rate", "Poor or fair health % Fair or Poor Health", "Poor physical health days Average Number of Physically Unhealthy Days", 
"Poor mental health days Average Number of Mentally Unhealthy Days", "Low birthweight % Low birthweight",  "Sexually transmitted infections Chlamydia Rate","Mammography screening % With Annual Mammogram", 
"Flu vaccinations % Vaccinated","COVID-19 age-adjusted mortality COVID-19 death rate", "Life expectancy Life Expectancy", "Premature age-adjusted mortality Age-adjusted Death Rate", "Child mortality Child Mortality Rate", 
"Infant mortality Infant Mortality Rate", "Frequent physical distress % Frequent Physical Distress", "Frequent mental distress % Frequent Mental Distress", "Diabetes prevalence % Adults with Diabetes", "HIV prevalence HIV Prevalence Rate",
"Drug overdose deaths Drug Overdose Mortality Rate",]

social_cols = ["Adult smoking % Smokers", "Adult obesity % Adults with Obesity", "Food environment index Food Environment Index", "Physical inactivity % Physically Inactive", 
"Access to exercise opportunities % With Access to Exercise Opportunities","Excessive drinking % Excessive Drinking", "Alcohol-impaired driving deaths % Driving Deaths with Alcohol Involvement", 
"Teen births Teen Birth Rate", "Uninsured % Uninsured", "Primary care physicians Primary Care Physicians Rate", "Dentists Dentist Rate", "Mental health providers Mental Health Provider Rate",
"High school completion % Completed High School", "Some college % Some College", "Unemployment % Unemployed", "Children in poverty % Children in Poverty", "Income inequality Income Ratio", 
"Children in single-parent households % Children in Single-Parent Households", "Social associations Social Association Rate", "Violent crime Violent Crime Rate", "Injury deaths Injury Death Rate", 
"Air pollution - particulate matter Average Daily PM2.5", "Severe housing problems % Severe Housing Problems", "Driving alone to work % Drive Alone to Work", "Long commute - driving alone % Long Commute - Drives Alone"
"Food insecurity % Food Insecure", "Limited access to healthy foods % Limited Access to Healthy Foods","Motor vehicle crash deaths Motor Vehicle Mortality Rate", "Insufficient sleep % Insufficient Sleep", 
"Uninsured adults % Uninsured", "Uninsured children % Uninsured","Other primary care providers Other Primary Care Provider Rate", "High school graduation High School Graduation Rate", "Disconnected youth % Disconnected Youth", 
"Reading scores Average Grade Performance", "Math scores Average Grade Performance", "School segregation Segregation index", "School funding adequacy School funding", "Gender pay gap Gender Pay Gap", "Median household income Median Household Income", 
"Children eligible for free or reduced price lunch % Enrolled in Free or Reduced Lunch", "Residential segregation - Black/white Segregation index", "Residential segregation - non-white/white Segregation Index", 
"Childcare cost burden % household income required for childcare expenses", "Childcare centers County Value", "Homicides Homicide Rate", "Suicides Suicide Rate (Age-Adjusted)", "Firearm fatalities Firearm Fatalities Rate", 
"Juvenile arrests Juvenile Arrest Rate", "Traffic volume Traffic Volume", "Homeownership % Homeowners", "Severe housing cost burden % Severe Housing Cost Burden", "Broadband access % Broadband Access", "Population Population", 
"% below 18 years of age % Less Than 18 Years of Age", "% 65 and older % 65 and Over", "% non-Hispanic Black % Black", "% American Indian & Alaska Native % American Indian & Alaska Native", "% Asian % Asian", 
"% Native Hawaiian/Other Pacific Islander % Native Hawaiian/Other Pacific Islander", "% Hispanic % Hispanic", "% non-Hispanic white % Non-Hispanic white", "% not proficient in English % Not Proficient in English", "% female % female", "% rural % rural"]

medical_df = df_no_outliers.loc[:, medical_cols]
social_df = df_no_outliers.loc[:, social_cols]


