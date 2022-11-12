import pandas as pd



def impute(df_ranked_measure):
    impute_cols = ["Premature death Years of Potential Life Lost Rate", "Poor or fair health % Fair or Poor Health", "Poor physical health days Average Number of Physically Unhealthy Days", "Poor mental health days Average Number of Mentally Unhealthy Days", "Low birthweight % Low birthweight", "Adult smoking % Smokers", "Adult obesity % Adults with Obesity", "Food environment index Food Environment Index", "Physical inactivity % Physically Inactive", "Access to exercise opportunities % With Access to Exercise Opportunities", "Excessive drinking % Excessive Drinking", "Alcohol-impaired driving deaths % Driving Deaths with Alcohol Involvement", "Sexually transmitted infections Chlamydia Rate", "Teen births Teen Birth Rate", "Uninsured % Uninsured", "Primary care physicians Primary Care Physicians Rate", "Dentists Dentist Rate", "Mental health providers Mental Health Provider Rate", "Preventable hospital stays Preventable Hospitalization Rate", "Mammography screening % With Annual Mammogram", "Flu vaccinations % Vaccinated", "High school completion % Completed High School", "Some college % Some College", "Unemployment % Unemployed", "Children in poverty % Children in Poverty", "Income inequality Income Ratio", "Children in single-parent households % Children in Single-Parent Households", "Social associations Social Association Rate", "Violent crime Violent Crime Rate", "Injury deaths Injury Death Rate", "Air pollution - particulate matter Average Daily PM2.5", "Severe housing problems % Severe Housing Problems", "Driving alone to work % Drive Alone to Work", "Long commute - driving alone % Long Commute - Drives Alone"
    , "COVID-19 age-adjusted mortality COVID-19 death rate", "Life expectancy Life Expectancy", "Premature age-adjusted mortality Age-adjusted Death Rate", "Child mortality Child Mortality Rate", "Infant mortality Infant Mortality Rate", "Frequent physical distress % Frequent Physical Distress", "Frequent mental distress % Frequent Mental Distress", "Diabetes prevalence % Adults with Diabetes", "HIV prevalence HIV Prevalence Rate", "Food insecurity % Food Insecure", "Limited access to healthy foods % Limited Access to Healthy Foods", "Drug overdose deaths Drug Overdose Mortality Rate", "Motor vehicle crash deaths Motor Vehicle Mortality Rate", "Insufficient sleep % Insufficient Sleep", "Uninsured adults % Uninsured", "Uninsured children % Uninsured", "Other primary care providers Other Primary Care Provider Rate", "High school graduation High School Graduation Rate", "Disconnected youth % Disconnected Youth", "Reading scores Average Grade Performance", "Math scores Average Grade Performance", "School segregation Segregation index", "School funding adequacy School funding", "Gender pay gap Gender Pay Gap", "Median household income Median Household Income", "Children eligible for free or reduced price lunch % Enrolled in Free or Reduced Lunch", "Residential segregation - Black/white Segregation index", "Residential segregation - non-white/white Segregation Index", "Childcare cost burden % household income required for childcare expenses", "Childcare centers County Value", "Homicides Homicide Rate", "Suicides Suicide Rate (Age-Adjusted)", "Firearm fatalities Firearm Fatalities Rate", "Juvenile arrests Juvenile Arrest Rate", "Traffic volume Traffic Volume", "Homeownership % Homeowners", "Severe housing cost burden % Severe Housing Cost Burden", "Broadband access % Broadband Access", "Population Population", "% below 18 years of age % Less Than 18 Years of Age", "% 65 and older % 65 and Over", "% non-Hispanic Black % Black", "% American Indian & Alaska Native % American Indian & Alaska Native", "% Asian % Asian", "% Native Hawaiian/Other Pacific Islander % Native Hawaiian/Other Pacific Islander", "% Hispanic % Hispanic", "% non-Hispanic white % Non-Hispanic white", "% not proficient in English % Not Proficient in English", "% female % female", "% rural % rural"]

    imputed_df = pd.DataFrame()
    imputed_df.reset_index()



    for i in impute_cols:
        imputed_series = df_ranked_measure.loc[:, i]
        median = df_ranked_measure[i].median()
        #print(f'median: {median}')
        imputed_series.fillna(median, inplace=True)
        imputed_df.assign(i=imputed_series)
        # imputed_df2 = imputed_series.to_frame()
        # imputed_df2.reset_index()
        #print(f'df2: {imputed_df2}')
        #imputed_df = imputed_df.merge(imputed_series.reset_index(), on=['index'])
        imputed_df = pd.concat([imputed_df, imputed_series], axis=1)

    return imputed_df