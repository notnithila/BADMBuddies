import pandas as pd


def impute():
    impute_cols = ["Years of Potential Life Lost Rate", "% Fair or Poor Health", "Average Number of Physically Unhealthy Days", "Average Number of Mentally Unhealthy Days", "% Low birthweight", "% Smokers", "% Adults with Obesity", "Food Environment Index", "% Physically Inactive", "% With Access to Exercise Opportunities", "% Excessive Drinking", "% Driving Deaths with Alcohol Involvement", "Chlamydia Rate", "Teen Birth Rate", "% Uninsured", "Primary Care Physicians Rate", "Dentist Rate", "Mental Health Provider Rate", "Preventable Hospitalization Rate", "% With Annual Mammogram", "% Vaccinated", "% Completed High School", "% Some College", "% Unemployed", "% Children in Poverty", "Income Ratio", "% Children in Single-Parent Households", "Social Association Rate", "Violent Crime Rate", "Injury Death Rate", "Average Daily PM2.5", "% Severe Housing Problems", "% Drive Alone to Work", "% Long Commute - Drives Alone"]

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