import pandas as pd
from impute.py import impute


# read excel file sheets into separate df
excel_path = '/Users/sheetalsudhir/Documents/BADMBuddies/data/2022_County_Health_Rankings_Data.xlsx'
df_outcomes_rankings = pd.read_excel(excel_path, "Outcomes & Factors Rankings")
df_outcomes_subrankings = pd.read_excel(excel_path, "Outcomes & Factors SubRankings")
df_ranked_measure = pd.read_excel(excel_path, "Ranked Measure Data")
df_additional_measure = pd.read_excel(excel_path, "Additional Measure Data")

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

df_outcomes_rankings = apply_rename_columns(df_outcomes_rankings)
df_outcomes_subrankings = apply_rename_columns(df_outcomes_subrankings)
df_ranked_measure = apply_rename_columns(df_ranked_measure)
df_additional_measure = apply_rename_columns(df_additional_measure)

massive_df = df_outcomes_rankings.merge(df_outcomes_subrankings, on=[" FIPS"," County"," State"])
massive_df = massive_df.merge(df_ranked_measure, on=[" FIPS"," County"," State"])
massive_df = massive_df.merge(df_additional_measure, on=[" FIPS"," County"," State"])

def impute(df):
    impute_cols = ["Premature death Years of Potential Life Lost Rate", "Poor or fair health % Fair or Poor Health", "Poor physical health days Average Number of Physically Unhealthy Days", "Poor mental health days Average Number of Mentally Unhealthy Days", "Low birthweight % Low birthweight", "Adult smoking % Smokers", "Adult obesity % Adults with Obesity", "Food environment index Food Environment Index", "Physical inactivity % Physically Inactive", "Access to exercise opportunities % With Access to Exercise Opportunities", "Excessive drinking % Excessive Drinking", "Alcohol-impaired driving deaths % Driving Deaths with Alcohol Involvement", "Sexually transmitted infections Chlamydia Rate", "Teen births Teen Birth Rate", "Uninsured % Uninsured", "Primary care physicians Primary Care Physicians Rate", "Dentists Dentist Rate", "Mental health providers Mental Health Provider Rate", "Preventable hospital stays Preventable Hospitalization Rate", "Mammography screening % With Annual Mammogram", "Flu vaccinations % Vaccinated", "High school completion % Completed High School", "Some college % Some College", "Unemployment % Unemployed", "Children in poverty % Children in Poverty", "Income inequality Income Ratio", "Children in single-parent households % Children in Single-Parent Households", "Social associations Social Association Rate", "Violent crime Violent Crime Rate", "Injury deaths Injury Death Rate", "Air pollution - particulate matter Average Daily PM2.5", "Severe housing problems % Severe Housing Problems", "Driving alone to work % Drive Alone to Work", "Long commute - driving alone % Long Commute - Drives Alone"
    , "COVID-19 age-adjusted mortality COVID-19 death rate", "Life expectancy Life Expectancy", "Premature age-adjusted mortality Age-adjusted Death Rate", "Child mortality Child Mortality Rate", "Infant mortality Infant Mortality Rate", "Frequent physical distress % Frequent Physical Distress", "Frequent mental distress % Frequent Mental Distress", "Diabetes prevalence % Adults with Diabetes", "HIV prevalence HIV Prevalence Rate", "Food insecurity % Food Insecure", "Limited access to healthy foods % Limited Access to Healthy Foods", "Drug overdose deaths Drug Overdose Mortality Rate", "Motor vehicle crash deaths Motor Vehicle Mortality Rate", "Insufficient sleep % Insufficient Sleep", "Uninsured adults % Uninsured", "Uninsured children % Uninsured", "Other primary care providers Other Primary Care Provider Rate", "High school graduation High School Graduation Rate", "Disconnected youth % Disconnected Youth", "Reading scores Average Grade Performance", "Math scores Average Grade Performance", "School segregation Segregation index", "School funding adequacy School funding", "Gender pay gap Gender Pay Gap", "Median household income Median Household Income", "Children eligible for free or reduced price lunch % Enrolled in Free or Reduced Lunch", "Residential segregation - Black/white Segregation index", "Residential segregation - non-white/white Segregation Index", "Childcare cost burden % household income required for childcare expenses", "Childcare centers County Value", "Homicides Homicide Rate", "Suicides Suicide Rate (Age-Adjusted)", "Firearm fatalities Firearm Fatalities Rate", "Juvenile arrests Juvenile Arrest Rate", "Traffic volume Traffic Volume", "Homeownership % Homeowners", "Severe housing cost burden % Severe Housing Cost Burden", "Broadband access % Broadband Access", "Population Population", "% below 18 years of age % Less Than 18 Years of Age", "% 65 and older % 65 and Over", "% non-Hispanic Black % Black", "% American Indian & Alaska Native % American Indian & Alaska Native", "% Asian % Asian", "% Native Hawaiian/Other Pacific Islander % Native Hawaiian/Other Pacific Islander", "% Hispanic % Hispanic", "% non-Hispanic white % Non-Hispanic white", "% not proficient in English % Not Proficient in English", "% female % female", "% rural % rural"]

    imputed_df = pd.DataFrame()
    imputed_df.reset_index()

    for i in impute_cols:
        imputed_series = df.loc[:, i]
        median = df[i].median()
        #print(f'median: {median}')
        imputed_series.fillna(median, inplace=True)
        imputed_df.assign(i=imputed_series)
        imputed_df = pd.concat([imputed_df, imputed_series], axis=1)

    return imputed_df

final_df = impute(massive_df)

# remove outliers
import numpy as np
from scipy import stats
temp = final_df[(np.abs(stats.zscore(final_df)) < 3).all(axis=1)]

# split dataframe into medical and social variables
df_no_outliers = temp.copy()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
scaled_values = scaler.fit_transform(df_no_outliers) 
df_no_outliers.loc[:,:] = scaled_values

medical_cols = ['Preventable hospital stays Preventable Hospitalization Rate', "Premature death Years of Potential Life Lost Rate", "Poor or fair health % Fair or Poor Health", "Poor physical health days Average Number of Physically Unhealthy Days", 
"Poor mental health days Average Number of Mentally Unhealthy Days", "Low birthweight % Low birthweight",  "Sexually transmitted infections Chlamydia Rate","Mammography screening % With Annual Mammogram", 
"Flu vaccinations % Vaccinated","COVID-19 age-adjusted mortality COVID-19 death rate", "Life expectancy Life Expectancy", "Premature age-adjusted mortality Age-adjusted Death Rate", "Child mortality Child Mortality Rate", 
"Infant mortality Infant Mortality Rate", "Frequent physical distress % Frequent Physical Distress", "Frequent mental distress % Frequent Mental Distress", "Diabetes prevalence % Adults with Diabetes", "HIV prevalence HIV Prevalence Rate",
"Drug overdose deaths Drug Overdose Mortality Rate",]

social_cols = ['Preventable hospital stays Preventable Hospitalization Rate', "Adult smoking % Smokers", "Adult obesity % Adults with Obesity", "Food environment index Food Environment Index", "Physical inactivity % Physically Inactive", 
"Access to exercise opportunities % With Access to Exercise Opportunities","Excessive drinking % Excessive Drinking", "Alcohol-impaired driving deaths % Driving Deaths with Alcohol Involvement", 
"Teen births Teen Birth Rate", "Uninsured % Uninsured", "Primary care physicians Primary Care Physicians Rate", "Dentists Dentist Rate", "Mental health providers Mental Health Provider Rate",
"High school completion % Completed High School", "Some college % Some College", "Unemployment % Unemployed", "Children in poverty % Children in Poverty", "Income inequality Income Ratio", 
"Children in single-parent households % Children in Single-Parent Households", "Social associations Social Association Rate", "Violent crime Violent Crime Rate", "Injury deaths Injury Death Rate", 
"Air pollution - particulate matter Average Daily PM2.5", "Severe housing problems % Severe Housing Problems", "Driving alone to work % Drive Alone to Work",
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

import seaborn as sns
# Assumption 1: Linearity
p = sns.pairplot(medical_df, x_vars=medical_cols, y_vars='Preventable hospital stays Preventable Hospitalization Rate', size=5, aspect=0.7)


# Perform regression

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


X_med = medical_df.drop('Preventable hospital stays Preventable Hospitalization Rate',axis= 1)
y_med = medical_df['Preventable hospital stays Preventable Hospitalization Rate']

X_med_train, X_med_test, y_med_train, y_med_test = train_test_split(X_med, y_med, test_size=0.2, 
                                                                    random_state=101)

medical_model = LinearRegression()
medical_model.fit(X_med_train, y_med_train)
medical_predictions = medical_model.predict(X_med_test)
medical_predictions_train = medical_model.predict(X_med_train)

print('mean_squared_error medical model:', mean_squared_error(y_med_test, medical_predictions))
print('mean_absolute_error medical model:', mean_absolute_error(y_med_test, medical_predictions))
print("R squared: {}\n".format(r2_score(y_true=y_med_test, y_pred=medical_predictions)))

# create feature variables
X_social = social_df.drop('Preventable hospital stays Preventable Hospitalization Rate',axis= 1)
y_social = social_df['Preventable hospital stays Preventable Hospitalization Rate']

X_social_train, X_social_test, y_social_train, y_social_test = train_test_split(X_social, y_social, test_size=0.2, 
                                                                                random_state=101)
social_model = LinearRegression()
social_model.fit(X_social_train, y_social_train)
social_predictions = social_model.predict(X_social_test)
social_predictions_train = social_model.predict(X_social_train)

print('mean_squared_error social model:', mean_squared_error(y_social_test, social_predictions))
print('mean_absolute_error social model:', mean_absolute_error(y_social_test, social_predictions))
print("R squared: {}\n".format(r2_score(y_true=y_social_test, y_pred=social_predictions)))

# Assumption 2
residuals_medical = y_med_train.values-medical_predictions_train
mean_residuals_medical = np.mean(residuals_medical)
print("Mean of Residuals - Medical DF: {}".format(mean_residuals_medical))

residuals_social = y_social_train.values-social_predictions_train
mean_residuals_social = np.mean(residuals_social)
print("Mean of Residuals - Social DF: {}".format(mean_residuals_social))

# Assumption 3 - Homoscedasticity

p = sns.scatterplot(medical_predictions_train, residuals_medical)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Medical Residuals')
plt.ylim(-2.5,2.5)
plt.xlim(0,1)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')

p = sns.scatterplot(social_predictions_train, residuals_social)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Social Residuals')
plt.ylim(-2.5,2.5)
plt.xlim(0,1)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')

# Permutation Feature Importance
from sklearn.inspection import permutation_importance
med_r = permutation_importance(medical_model, X_med_train, y_med_train,
                            n_repeats=30,
                            random_state=0)

for i in med_r.importances_mean.argsort()[::-1]:
     if med_r.importances_mean[i] - 2 * med_r.importances_std[i] > 0:
        print(f"{X_med_test.columns[i]:<8}  "
              f"{med_r.importances_mean[i]:.3f}"
              f" +/- {med_r.importances_std[i]:.3f}")

# Permutation Feature Importance
from sklearn.inspection import permutation_importance
med_r = permutation_importance(social_model, X_social_test, y_social_test,
                            n_repeats=30,
                            random_state=0)

for i in med_r.importances_mean.argsort()[::-1]:
     if med_r.importances_mean[i] - 2 * med_r.importances_std[i] > 0:
        print(f"{X_social_test.columns[i]:<8}  "
              f"{med_r.importances_mean[i]:.3f}"
              f" +/- {med_r.importances_std[i]:.3f}")