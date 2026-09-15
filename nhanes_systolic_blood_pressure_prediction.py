"""
Factors Associated with Systolic Blood Pressure

Dataset:
NHANES 2017-2018

Objective:
Investigate factors associated with systolic blood pressure and examine multiple linear regression assumptions.

Models:
Baseline and reduced multiple linear regression

Evaluation:
RMSE, MAE, R²; residual diagnostics

Key Result:
Baseline and reduced models showed modest predictive performance, while multicollinearity substantially affected some
coefficient estimates.
"""


# 1. Import & Settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)


# 2. Data Loading & Dataset Construction

# 2.1. Demographics: Define it & select potential variables that might be associated with blood pressure.

demographics = pd.read_sas("demographics.xpt")

selected_demographics = demographics[[
    'SEQN',
    'RIAGENDR',
    'RIDAGEYR',
    'RIDRETH3',
    'DMDBORN4',
    'RIDEXPRG',
    'INDFMPIR',
    'DMDEDUC2'
]].round(2)

# Remove pregnancy status because it only applies to a small group of participants.
selected_demographics = selected_demographics.drop(columns='RIDEXPRG')

# Remove education to avoid losing many participants (~40%) in later analyses.
selected_demographics = selected_demographics.drop(columns='DMDEDUC2')

# Rename variables for readability.
selected_demographics = selected_demographics.rename(columns={
    'RIAGENDR':'Sex',
    'RIDAGEYR':'Age',
    'RIDRETH3':'RaceEthnicity',
    'DMDBORN4':'CountryOfBirth',
    'INDFMPIR':'IncomePovertyRatio',
})

# Replace category codes with readable labels. They will be encoded later before building the model.
selected_demographics['Sex'] = selected_demographics['Sex'].replace({
    1: 'Male',
    2: 'Female'
})

selected_demographics["RaceEthnicity"] = selected_demographics["RaceEthnicity"].replace({
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multi-Racial"
})

selected_demographics['CountryOfBirth'] = selected_demographics['CountryOfBirth'].replace({
    1: 'US or Washington,DC',
    2: 'Others',
    77: np.nan,
    99: np.nan
})

# Restrict the study population to adults (18 years and older).
selected_demographics = selected_demographics[selected_demographics['Age'] >= 18]


# 2.2. Blood pressure(target): Define it & use just the first three systolic measurements since the fourth one has
# too many missing values.

blood_pressure = pd.read_sas("blood_pressure.xpt")

selected_blood_pressure = blood_pressure[[
    'SEQN',
    'BPXSY1',
    'BPXSY2',
    'BPXSY3'
]].round(2)

# Remove participants without any systolic blood pressure measurements.
mask = selected_blood_pressure.isna().sum(axis=1) < 3
selected_blood_pressure = selected_blood_pressure[mask]

# Use the mean of the available systolic measurements to reduce random measurement variation.
selected_blood_pressure['Systolic_BP_Mean'] = (
    selected_blood_pressure[['BPXSY1', 'BPXSY2', 'BPXSY3']]
    .mean(axis=1)
    .round(2)
)

selected_blood_pressure = selected_blood_pressure[[
    'SEQN',
    'Systolic_BP_Mean'
]]

# Merge blood pressure data with the adult demographic population.
merged_df = pd.merge(
    selected_blood_pressure,
    selected_demographics,
    on='SEQN',
    how='inner'
)


# 2.3. Alcohol use: Define it & select alcohol history and drinking frequency in the last 12 months.

alcohol_use = pd.read_sas("alcohol_use.xpt")

selected_alcohol_use = alcohol_use[[
    'SEQN',
    'ALQ111',
    'ALQ121',
]].round(2)

# Rename variables for readability.
selected_alcohol_use = selected_alcohol_use.rename(columns={
    'ALQ111': 'EverAlcohol',
    'ALQ121': 'AlcoholFrequency'
})

# Replace category codes with readable labels. They will be encoded later before building the model.
selected_alcohol_use['EverAlcohol'] = selected_alcohol_use['EverAlcohol'].replace({
    1: 'Yes',
    2: 'No',
    7: np.nan,
    9: np.nan
})

selected_alcohol_use['AlcoholFrequency'] = (
    pd.to_numeric(selected_alcohol_use['AlcoholFrequency'], errors='coerce')
    .map({
        0: 'Never in the last year',
        1: 'Every day',
        2: 'Nearly every day',
        3: '3-4 times/week',
        4: '2 times/week',
        5: 'Once/week',
        6: '2-3 times/month',
        7: 'Once/month',
        8: '7-11 times/year',
        9: '3-6 times/year',
        10: '1-2 times/year',
        77: np.nan,
        99: np.nan
    })
)

# Merge alcohol variables.
merged_df = pd.merge(
    merged_df,
    selected_alcohol_use,
    on='SEQN',
    how='left'
)


# 2.4. Smoking: Define it & select smoking history, current smoking status, and cigarettes per day.

smoking = pd.read_sas("smoking.xpt")

selected_smoking = smoking[[
    'SEQN',
    'SMQ020',
    'SMQ040',
    'SMD650'
]].round(2)

# Rename variables for readability and replace category codes with readable labels. They will be encoded later before
# building the model.
selected_smoking['Smokedatleast100ciginlife'] = selected_smoking['SMQ020'].replace({
    1: 'Yes',
    2: 'No',
    7: np.nan,
    9: np.nan
})

selected_smoking['CurrentSmoking'] = selected_smoking['SMQ040'].replace({
    1: 'Every day',
    2: 'Some days',
    3: 'Not at all',
    7: np.nan,
    9: np.nan
})

selected_smoking['CigarettesPerDay'] = selected_smoking['SMD650'].replace({
    777: np.nan,
    999: np.nan
})

selected_smoking = selected_smoking.drop(columns=['SMQ020', 'SMQ040', 'SMD650'])

# Merge smoking variables.
merged_df = pd.merge(
    merged_df,
    selected_smoking,
    on='SEQN',
    how='left'
)


# 2.5. Physical activity: Define it & select intentional vigorous and moderate activity; also sedentary time.

physical_activity = pd.read_sas("physical_activity.xpt")

selected_physical_activity = physical_activity[[
    'SEQN',
    'PAQ650',
    'PAQ665',
    'PAD680'
]].round(2)

# Rename variables for readability and replace category codes with readable labels. They will be encoded later before
# building the model.
selected_physical_activity['VigorousActivity'] = selected_physical_activity['PAQ650'].replace({
    1: 'Yes',
    2: 'No'
})

selected_physical_activity['ModerateActivity'] = selected_physical_activity['PAQ665'].replace({
    1: 'Yes',
    2: 'No'
})

selected_physical_activity = selected_physical_activity.rename(columns={
    'PAD680': 'SedentaryActivityinMins'
})

# Replace answer 9999 (i.e. 'Don't know') with actual NaN values. It will be handled in preprocessing.
selected_physical_activity['SedentaryActivityinMins'] = (
    selected_physical_activity['SedentaryActivityinMins']
    .replace(9999, np.nan)
)

selected_physical_activity = selected_physical_activity.drop(columns=['PAQ650', 'PAQ665'])

# Merge physical activity variables.
merged_df = pd.merge(
    merged_df,
    selected_physical_activity,
    on='SEQN',
    how='left'
)


# 2.6. Body measures: Define it & select BMI and waist circumference.
# BMI for providing general body size/adiposity information.
# Waist circumference for providing central adiposity which BMI does not distinguish.

body_measures = pd.read_sas("body_measures.xpt")

selected_body_measures = body_measures[[
    'SEQN',
    'BMXBMI',
    'BMXWAIST'
]]

# Rename variables for readability.
selected_body_measures = selected_body_measures.rename(columns={
    'BMXBMI': 'BMI(kg/m**2)',
    'BMXWAIST': 'WaistCircumference(cm)',
})

# Merge body measure variables.
merged_df = pd.merge(
    merged_df,
    selected_body_measures,
    on='SEQN',
    how='left'
)


# 2.7. Diabetes status: Define it & select self-reported diabetes status.

diabetes_status = pd.read_sas("diabetes_status.xpt")

selected_diabetes_status = diabetes_status[[
    'SEQN',
    'DIQ010'
]]

# Rename the variable for readability and replace category codes with readable labels. They will be encoded later
# before building the model.
selected_diabetes_status['DiabetesStatus'] = selected_diabetes_status['DIQ010'].replace({
    1: 'Yes',
    2: 'No',
    3: 'Borderline',
    7: np.nan,
    9: np.nan
})

selected_diabetes_status = selected_diabetes_status.drop(columns=['DIQ010'])

# Merge diabetes status variable.
merged_df = pd.merge(
    merged_df,
    selected_diabetes_status,
    on='SEQN',
    how='left'
)


# 2.8. Hypertension: Define it & select hypertension history and current hypertension medication status.

hypertension = pd.read_sas("bloodpressure_and_cholesterol.xpt")

selected_hypertension = hypertension[[
    'SEQN',
    'BPQ020',
    'BPQ050A'
]]

# Rename variables for readability and replace category codes with readable labels. They will be encoded later before
# building the model.
selected_hypertension['HypertensionHistory'] = (
    selected_hypertension['BPQ020'].replace({
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    })
)

selected_hypertension['HypertensionMedication'] = (
    selected_hypertension['BPQ050A'].replace({
        1: 'Yes',
        2: 'No',
        7: np.nan,
        9: np.nan
    })
)

selected_hypertension = selected_hypertension.drop(columns=['BPQ020', 'BPQ050A'])

# Merge hypertension variables.
merged_df = pd.merge(
    merged_df,
    selected_hypertension,
    on='SEQN',
    how='left'
)


# 2.9. Fasting glucose: Define it & select the fasting glucose measurement in mg/dL.

fasting_glucose = pd.read_sas("fasting_glucose.xpt")

selected_fasting_glucose = fasting_glucose[[
    'SEQN',
    'LBXGLU'
]]

# Rename the variable and replace entries '0' with NaN in fasting glucose measurement because they mean either no lab
# result or not fasting 8 to <24 hrs.
selected_fasting_glucose['FastingGlucose(mg/dL)'] = selected_fasting_glucose['LBXGLU'].replace(0, np.nan)

selected_fasting_glucose = selected_fasting_glucose.drop(columns=['LBXGLU'])

# Merge fasting glucose variable.
merged_df = pd.merge(
    merged_df,
    selected_fasting_glucose,
    on='SEQN',
    how='left'
)


# 2.10. Total cholesterol: Define it & select the total cholesterol measurement in mg/dL.

total_cholesterol = pd.read_sas("total_cholesterol.xpt")

selected_total_cholesterol = total_cholesterol[[
    'SEQN',
    'LBXTC'
]]

# Rename the variable.
selected_total_cholesterol = selected_total_cholesterol.rename(columns={
    'LBXTC': 'TotalCholesterol(mg/dL)',
})

# Merge total cholesterol variable.
merged_df = pd.merge(
    merged_df,
    selected_total_cholesterol,
    on='SEQN',
    how='left'
)


# 3. Exploratory Data Analysis

df = merged_df.copy()

# Define numerical and categorical variables
numerical_variables = df.select_dtypes(include='number').drop(columns=['SEQN'])
categorical_variables = df.select_dtypes(include='object')
print(f"Number of numerical variables: {len(numerical_variables.columns)}")
print(f"Number of categorical variables: {len(categorical_variables.columns)}")


# 3.1. Numerical variables - descriptive stats, distribution, outliers, and findings

for variable in numerical_variables.columns:
    missing_count = numerical_variables[variable].isna().sum()
    missing_percentage = missing_count / len(numerical_variables[variable]) *100
    print('=' * 65)
    print(f'Number of missing values for {variable}: {missing_count}')
    print(f'Percentage of missing values for {variable}: {missing_percentage.round(2)}%')

    print(f'Statistics for {variable}:')
    print(numerical_variables[variable].describe().round(2))

    plt.figure()
    numerical_variables[variable].plot(kind='hist')
    plt.title(f'{variable} Distribution')
    plt.ylabel('Frequency')
    plt.xlabel(variable)
    plt.show()

    plt.figure()
    numerical_variables[variable].plot(kind='box', showmeans=True)
    plt.title(f'{variable} Boxplot')
    plt.ylabel(variable)
    plt.show()

    Q1 = numerical_variables[variable].quantile(0.25)
    Q3 = numerical_variables[variable].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = numerical_variables[
        (numerical_variables[variable] < lower_bound) |
        (numerical_variables[variable] > upper_bound)
    ][variable]

    valid_count = numerical_variables[variable].notna().sum()

    print(f'Number of outliers: {outliers.notna().sum()}')
    print(f'Percentage of outliers: {(outliers.notna().sum() / valid_count) * 100:.2f}%')
    print()

# Potential outliers were identified using the 1.5×IQR rule. They were not automatically removed because extreme
# physiological measurements may represent valid observations.

# Systolic_BP_Mean is right-skewed, with 149 (2.84%) observations identified as potential outliers. These observations
# were retained because they may represent valid high blood pressure measurements.

# Age is restricted to adult participants (>= 18). The mean age is ~50 years and the median is 51 years, with
# ages top-coded at 80 in NHANES.

# IncomePovertyRatio is capped at 5.0 by NHANES design. The distribution is positively skewed, with the mean (2.53)
# higher than the median (2.11), and there is a large concentration of values at the upper cap. No potential outliers
# are detected.

# CigarettesPerDay is right-skewed, with the mean (11.07) higher than the median (10.00). There are 16 (1.76%) potential
# outliers. The variable has 82.66% missing values, largely due to questionnaire skip patterns. Missing-value handling
# is addressed separately in the missing-data section.

# SedentaryActivityinMins is positively skewed, with mean (332.41) larger than median (300.00). There are 45 potential
# outliers (0.86%) and 38 missing values (0.72%).

# BMI is right-skewed, with the mean (29.67) higher than the median (28.40). There are 147 (2.84%) potential outliers
# and 71 (1.35%) missing values.

# WaistCircumference(cm) is positively skewed, with the mean (100.15) larger than the median (98.90). Number of the
# potential outliers is 61 (1.22%) and number of the missing entries is 252 (4.8%).

# FastingGlucose(mg/dL) is right-skewed, with the mean of 113.89 being greater than the median of 104.00. There are
# 256 (10.62%) potential outliers and high number of missing data as 2836 (54.05%), resulting from unavailable fasting
# glucose measurements due to no lab result or insufficient fasting duration.

# TotalCholesterol(mg/dL) is right skewed, with the mean of 187.09 being higher than the median of 183.00.
# Number of the potential outliers is 68 (1.38%) and missing entries is 322 (6.14%).


# 3.2. Categorical variables - frequencies, proportions, distributions, and findings

for variable in categorical_variables.columns:
    missing_count = categorical_variables[variable].isna().sum()
    missing_percentage = missing_count / len(categorical_variables[variable]) * 100
    frequencies = categorical_variables[variable].value_counts(dropna=False)
    percentages = categorical_variables[variable].value_counts(
        normalize=True,
        dropna=False
    ) * 100
    summary = pd.DataFrame({
        'Frequency': frequencies,
        'Percentage': percentages.round(2)
    })

    print(f'Variable: {variable}')
    print(f'Missing values: {missing_count} ({missing_percentage:.2f}%)')
    print(summary)
    print('=' * 50)

    plt.figure()
    categorical_variables[variable].value_counts(dropna=False).plot(kind='bar')
    plt.title(f'{variable} Barplot')
    plt.ylabel('Frequency')
    plt.xlabel(variable)
    plt.tight_layout()
    plt.show()

# Sex is almost evenly distributed with 51.46% female and 48.54% male participants. There are no missing values.

# RaceEthnicity has six categories with 'Non-Hispanic White' being the most common (34.67%) and 'Other/Multi-Racial'
# being the least common (5.30%). There are no missing values.

# CountryOfBirth has two categories which are 'US or Washington, DC' and 'Others'. First one dominates with ~70% while
# second one is about 30%. There are two missing values (0.04 %), which is negligible.

# EverAlcohol has two categories: 'Yes' and 'No'. 'Yes' is predominant with ~84% and 'No' represents ~10% of the
# observations while there are 307 (5.85%) missing values, which represent responses that were unavailable or coded
# as missing.

# AlcoholFrequency (in the last 12 months) consists of 11 categories. 'Never in the last year' is the most common
# response (19.29%) while 'Every day' is the least common (2.59%). The remaining categories range between these values.
# There are 854 (16.28%) missing values, which is due to skip pattern in the study.

# Smokedatleast100ciginlife has two categories, 'Yes' (40.77%) and 'No' (59.23%). There are no missing values.

# CurrentSmoking has 3 categories. 'Not at all' is the most common response (23.16%), followed by 'Every day' (13.84%),
# and 'Some days' (3.77%). Even though there are 3108 missing values (59.23%), which is substantially high, the number
# matches with 'No' percentages in 'Smokedatleast100ciginlife' variable meaning that the missingness is structural due
# to the questionnaire skip pattern.

# VigorousActivity has two categories with 'No' representing 75.15% and 'Yes' representing 24.85% of observations.
# There are no missing values.

# ModerateActivity has two categories with 'No' representing 59.81% and 'Yes' representing 40.19% of observations.
# There are no missing values.

# DiabetesStatus has three categories. 'No' is predominant with 81.72%, followed by 'Yes' with 15.21%, and 'Borderline'
# with 2.99%. There are 4 missing values (0.08%) which is negligible.

# HypertensionHistory has two categories: 'No' represents 63.03% and 'Yes' represents 36.78% of observations. There are
# 10 missing values (0.19%) which is negligible.

# HypertensionMedication has two categories. 'Yes' represents 28.51% and 'No' represents 4.90% of the total sample.
# There are 3494 missing values (66.59%), which are structural missingness due to skip pattern in the questionnaire.


# 3.3. Relationship between feature matrix and target

target = 'Systolic_BP_Mean'

# 3.3.1. Numerical variables vs. target
numerical_variables_without_target = numerical_variables.drop(columns=[target])

print(f'Relationship (Pearson Correlation) between numerical variables:')
print('=' * 65)

for variable in numerical_variables_without_target.columns:
    print(f'{variable} and {target}: {df[variable].corr(df[target]):.3f}')
    print('-'*65)

    df.plot(
        kind='scatter',
        x=variable,
        y=target
    )
    plt.title(f'{variable} vs {target}')
    plt.xlabel(variable)
    plt.ylabel(target)
    plt.show()

# Based on Pearson correlation:

# Age and the target have a moderate positive linear association (0.487).

# Weak associations with the target are BMI(kg/m**2) with 0.153, WaistCircumference(cm) with 0.224,
# FastingGlucose(mg/dL) with 0.180, and TotalCholesterol(mg/dL) with 0.123.

# IncomePovertyRatio, CigarettesPerDay, and SedentaryActivityinMins have near-zero Pearson correlation values with the
# target, indicating little to no linear association.

# The scatter plots generally support the observed direction and strength of the Pearson correlations.


# 3.3.2. Categorical variables vs. target
for variable in categorical_variables.columns:
    print(f'\n===== {variable} =====')
    print(df.groupby(variable)['Systolic_BP_Mean'].describe())

    df.boxplot(column='Systolic_BP_Mean', by=variable, figsize=(12,8))
    plt.title(f'Systolic BP by {variable}')
    plt.suptitle('')
    plt.xlabel(variable)
    plt.ylabel('Systolic BP')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# The medians for 'Male' (124.67) and 'Female' (121.33) are close to each other. Means are also similar with 'Male'
# being 127.41 and 'Female' being '125.38'. 'Female' has greater variability with values of std 21.61 vs 18.37,
# in addition, IQR is also wider in 'Female' with 27.34 while for 'Male' is 22.00. The distributions substantially
# overlap, even though there is a difference in medians. Both groups include potential outliers.

# Non-Hispanic Black participants show noticeably higher systolic BP than the other groups, while the remaining groups
# have relatively similar distributions. Potential outliers are present across all groups.

# CountryOfBirth groups are broadly similar across the two groups, with substantial overlap. There is no notable
# difference in central tendency between the two groups. There are potential outliers for both groups.

# EverAlcohol groups are broadly similar across two groups, with substantial overlap. There is no notable difference
# in central tendency between two groups. There are potential outliers for both groups, with more visible potential
# outliers in the 'Yes' group.

# AlcoholFrequency groups show substantial overlap and are generally similar in systolic BP. 'Every day' and
# 'Never in the last year' show noticeably higher central tendency than most other groups. Potential outliers are
# present across all groups.

# Smokedatleast100ciginlife groups show substantial overlap and generally similar in systolic BP, although
# the 'Yes' group has a slightly higher central tendency. Both groups have potential outliers.

# CurrentSmoking groups show similarity in general for systolic BP, but 'Not at all' group has a slightly higher
# central tendency. All three groups have potential outliers.

# VigorousActivity shows differences for 'Yes' and 'No' groups. Lower central tendency and variability in 'Yes' group
# is noteworthy. Both groups include potential outliers.

# ModerateActivity shows differences for 'Yes' and 'No' groups. Slightly lower central tendency and variability in
# 'Yes' group is noteworthy. Both groups include potential outliers.

# DiabetesStatus indicates similar central tendency and variability in 'Yes' and 'Borderline' groups, while there
# is lower central tendency and slightly lower variability in 'No' group. Potential outliers are present in all
# three groups, with more visible potential outliers in the 'Yes' and 'No' groups.

# HypertensionHistory shows noticeable differences between 'Yes' and 'No' groups. The 'No' group has lower central
# tendency and variability. Both groups have potential outliers.

# HypertensionMedication groups show broadly similar central tendency with substantial overlap. The 'No' group has
# greater variability. Both groups include potential outliers.


# 3.4. Predictor relationships - Correlation matrix

corr_matrix = numerical_variables_without_target.corr(method='pearson')
print(corr_matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    vmax=1,
    vmin=-1,
    center=0,
    cmap='coolwarm',
    annot=True, fmt='.2f',
)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Convert upper-triangular correlation values to long format for sorting.
corr_stacked = (
    corr_matrix
    .where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    .stack()
    .reset_index()
    .dropna()
)

corr_stacked = corr_stacked.rename(columns={
    'level_0': 'variable 1',
    'level_1': 'variable 2',
    0: 'Correlation'
})

corr_stacked.sort_values(by='Correlation', key=abs, ascending=False)

# BMI(kg/m**2) and 'WaistCircumference(cm)' show strong positive linear association with r = 0.905. The remaining
# predictor pairs show relatively weak linear associations with |r| < 0.25. BMI and WaistCircumference may provide
# overlapping information and should be examined for potential multicollinearity.


# 4. Missing Data Analysis & Decisions

print(
    f'Missing value percentages:\n'
    f'{(df.isna().mean() * 100).sort_values(ascending=False)}'
)

# Missing values can be grouped into three categories:
# 1. Low-level non-structural missingness:
#    CountryOfBirth, DiabetesStatus, HypertensionHistory, SedentaryActivityinMins
# 2. Structural/questionnaire missingness:
#    CurrentSmoking, CigarettesPerDay, AlcoholFrequency, HypertensionMedication
# 3. Subsample missingness:
#    FastingGlucose


# 4.1. FastingGlucose(mg/dL)
# It will be excluded from the main regression model since ~54% of observations are missing due to the fasting
# laboratory subsample. Otherwise, it would substantially reduce available sample or require extensive imputation.
df = df.drop(columns=['FastingGlucose(mg/dL)'])


# 4.2. CigarettesPerDay
# It has mostly structural missingness. CurrentSmoking was only asked to participants who reported smoking at least
# 100 cigarettes in their lifetime. Missing CurrentSmoking values therefore correspond to participants who were not
# asked the question. Together with 'Not at all', these cases are represented as 0 in CigarettesPerDay.
df.loc[df['CurrentSmoking'].isna() | (df['CurrentSmoking'] == 'Not at all'),
       'CigarettesPerDay'] = 0


# 4.3. HypertensionMedication
# It has mostly structural missingness. The question was primarily asked to participants who reported 'Yes' to
# HypertensionHistory. Therefore, participants with HypertensionHistory = 'No' and missing medication status are
# assigned 'No'.
df.loc[df['HypertensionHistory'] == 'No', 'HypertensionMedication'] = 'No'


# 4.4. CurrentSmoking
# It has mostly structural missing data which is caused by the skip pattern of the questionnaire. Participants who
# answered 'Yes' to Smokedatleast100ciginlife were asked the 'CurrentSmoking' question. Participants with missing
# CurrentSmoking values are therefore treated as non-current smokers and assigned 'Not at all'.
df['CurrentSmoking'] = df['CurrentSmoking'].fillna('Not at all')


# 4.5. AlcoholFrequency
# It has mostly structural missingness caused by the questionnaire skip pattern. Participants who answered 'No' to
# EverAlcohol were not asked about drinking frequency and are assigned 'Never in the last year'.
df.loc[
    (df['EverAlcohol'] == 'No') & (df['AlcoholFrequency'].isna()),
    'AlcoholFrequency'
] = 'Never in the last year'

# Remaining missing values cannot be inferred confidently and are represented as 'Unknown'.
df['AlcoholFrequency'] = df['AlcoholFrequency'].fillna('Unknown')


# 4.6. IncomePovertyRatio

categorical_variables = df.select_dtypes(include='object')
numerical_variables = df.select_dtypes(include='number')
numerical_variables_without_target = numerical_variables.drop(columns=[target, 'SEQN'])

for variable in categorical_variables:
    print('=' * 65)
    print(f'IncomePovertyRatio missingness by {variable}:')

    missing_rate = (
        df.groupby(variable, dropna=False)['IncomePovertyRatio']
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(missing_rate)
# IncomePovertyRatio missingness is generally similar across most categorical groups, although noticeable differences
# are observed across RaceEthnicity and CountryOfBirth categories.

income_missing = np.where(df['IncomePovertyRatio'].isna(),
                          'Missing',
                          'Observed')

for variable in numerical_variables_without_target.drop(columns=['IncomePovertyRatio']):
    pd.DataFrame({
        variable: df[variable],
        'IncomeStatus': income_missing,
    }).boxplot(column=variable, by='IncomeStatus')

    plt.show()
# The distributions of the numerical predictors are broadly similar between participants with missing and observed
# IncomePovertyRatio.

# Overall, median imputation strategy is going to be used during model training.


# 4.7. TotalCholesterol(mg/dL)

for variable in categorical_variables:
    print('=' * 65)
    print(f'TotalCholesterol(mg/dL) missingness by {variable}:')

    missing_rate = (
        df.groupby(variable, dropna=False)['TotalCholesterol(mg/dL)']
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(missing_rate)
# TotalCholesterol missingness is generally similar across most categorical groups, although noticeable differences
# are observed across RaceEthnicity categories.

cholesterol_missing = np.where(
    df['TotalCholesterol(mg/dL)'].isna(),
    'Missing',
    'Observed'
)

for variable in numerical_variables_without_target.drop(columns=['TotalCholesterol(mg/dL)']):
    pd.DataFrame({
        variable: df[variable],
        'CholesterolStatus': cholesterol_missing,
    }).boxplot(column=variable, by='CholesterolStatus')

    plt.show()
# Age shows a noticeable difference between the missing and observed TotalCholesterol groups, with the missing group
# having a lower central tendency.

# Overall, TotalCholesterol missingness is not completely uniform across observed characteristics, particularly Age.


# 4.8. WaistCircumference(cm)

for variable in categorical_variables:
    print('=' * 65)
    print(f'WaistCircumference(cm) missingness by {variable}:')

    missing_rate = (
        df.groupby(variable, dropna=False)['WaistCircumference(cm)']
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(missing_rate)
# WaistCircumference missingness appears heterogeneous across observed characteristics.

waist_circumference_missing = np.where(
    df['WaistCircumference(cm)'].isna(),
    'Missing',
    'Observed'
)

for variable in numerical_variables_without_target.drop(columns=['WaistCircumference(cm)']):
    pd.DataFrame({
        variable: df[variable],
        'WaistCircumferenceStatus': waist_circumference_missing,
    }).boxplot(column=variable, by='WaistCircumferenceStatus')

    plt.show()
# Age, BMI, and IncomePovertyRatio show noteworthy differences in Missing vs Observed waist circumference.

# 4.8.1. WaistCircumference(cm) - inspecting Age
df_temp = pd.DataFrame({
    'Age': df['Age'],
    'WaistCircumferenceStatus': waist_circumference_missing
})

df_temp.groupby('WaistCircumferenceStatus')['Age'].describe()
# Age shows a noticeably higher central tendency in the Missing group, with slightly greater variability compared to
# the Observed group.

# 4.8.2. WaistCircumference(cm) - inspecting BMI
df_temp = pd.DataFrame({
    'BMI': df['BMI(kg/m**2)'],
    'WaistCircumferenceStatus': waist_circumference_missing
})

df_temp.groupby('WaistCircumferenceStatus')['BMI'].describe()
# BMI shows a higher central tendency, and greater variability in the Missing group compared with the Observed group.

# 4.8.3. WaistCircumference(cm) - inspecting IncomePovertyRatio
df_temp = pd.DataFrame({
    'IncomeStatus': df['IncomePovertyRatio'],
    'WaistCircumferenceStatus': waist_circumference_missing
})

df_temp.groupby('WaistCircumferenceStatus')['IncomeStatus'].describe()
# IncomePovertyRatio shows slightly lower central tendency, and slightly smaller variability in the Missing group
# compared with the Observed group.

# Overall, median imputation will be used for missing WaistCircumference during model training.


# 4.9. EverAlcohol

for variable in categorical_variables:
    print('=' * 55)
    print(f'EverAlcohol missingness by {variable}:')

    missing_rate = (
        df.groupby(variable, dropna=False)['EverAlcohol']
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(missing_rate)
# EverAlcohol missingness varies across several categorical groups, with more noticeable differences observed for
# RaceEthnicity, Sex, and CountryOfBirth.

ever_alcohol_missing = np.where(
    df['EverAlcohol'].isna(),
    'Missing',
    'Observed'
)

for variable in numerical_variables_without_target:
    pd.DataFrame({
        variable: df[variable],
        'EverAlcohol': ever_alcohol_missing,
    }).boxplot(column=variable, by='EverAlcohol')

    plt.show()
# IncomePovertyRatio, SedentaryActivityinMins, and WaistCircumference(cm) have lower central tendency in the Missing
# group compared with Observed group.

# 4.9.1. EverAlcohol - inspecting IncomePovertyRatio
df_temp = pd.DataFrame({
    'IncomeStatus': df['IncomePovertyRatio'],
    'EverAlcohol': ever_alcohol_missing
})

df_temp.groupby('EverAlcohol')['IncomeStatus'].describe()
# IncomePovertyRatio shows slightly lower central tendency, and lower variability in the Missing group compared
# with the Observed group.

# 4.9.2. EverAlcohol - inspecting SedentaryActivityinMins
df_temp = pd.DataFrame({
    'SedentaryActivity': df['SedentaryActivityinMins'],
    'EverAlcohol': ever_alcohol_missing
})

df_temp.groupby('EverAlcohol')['SedentaryActivity'].describe()
# SedentaryActivityinMins shows slightly lower central tendency, and lower variability in the Missing group compared
# with the Observed group.

# 4.9.3. EverAlcohol - inspecting WaistCircumference(cm)
df_temp = pd.DataFrame({
    'WaistCircumference': df['WaistCircumference(cm)'],
    'EverAlcohol': ever_alcohol_missing
})

df_temp.groupby('EverAlcohol')['WaistCircumference'].describe()
# WaistCircumference shows lower central tendency, and lower variability in the Missing group compared with the
# Observed group.

# Overall, EverAlcohol contains unresolved missing values that cannot be confidently inferred from other variables.
# Missing values will be represented as an explicit 'Unknown' category.
df['EverAlcohol'] = df['EverAlcohol'].fillna('Unknown')


# 4.10. HypertensionHistory, DiabetesStatus, and CountryOfBirth

# Missingness in these variables is negligible (<0.2% each). Complete-case removal has minimal impact on sample size
# and avoids imputing categorical values.
df = df.dropna(subset=['HypertensionHistory', 'DiabetesStatus', 'CountryOfBirth'])  # 16 rows are removed.


# 4.11. Decisions for remaining missing values

# Remaining missing values in AlcoholFrequency cannot be confidently inferred from the observed variables.
# They will be represented as an explicit 'Unknown' category.
df['AlcoholFrequency'] = df['AlcoholFrequency'].fillna('Unknown')

# Remaining missing values in HypertensionMedication cannot be confidently inferred from the observed variables.
# They will be represented as an explicit 'Unknown' category.
df['HypertensionMedication'] = df['HypertensionMedication'].fillna('Unknown')

# Remaining missing values in CigarettesPerDay will be filled with the median of observed non-zero values.
# The median will be learned from the training set only.

print(
    f'Missing value percentages:\n'
    f'{(df.isna().mean() * 100).sort_values(ascending=False)}'
)


# 5. Initial Feature Selection

# Demographic and socioeconomic variables:
# Sex and Age will be included because both are established factors associated with systolic blood pressure.
# IncomePovertyRatio will be included because socioeconomic conditions can affect lifestyle, diet, living conditions,
# and access to healthcare.
# RaceEthnicity will be included because differences related to ancestry, environmental, cultural, and social factors
# may be associated with blood pressure.
# CountryOfBirth will also be kept as a candidate because it may provide information about early-life environment,
# cultural background, and other factors not fully represented by RaceEthnicity or IncomePovertyRatio.

# Lifestyle variables:
# EverAlcohol and Smokedatleast100ciginlife will be excluded because they represent broader historical information,
# while AlcoholFrequency, CurrentSmoking, and CigarettesPerDay provide more specific information about alcohol and
# smoking behavior. AlcoholFrequency, CurrentSmoking, and CigarettesPerDay will be included in the model.
# SedentaryActivityinMins will also be included because sedentary behavior represents time spent sitting or reclining
# and is different from moderate or vigorous physical activity.
# VigorousActivity and ModerateActivity will be included because physical activity may be associated with systolic
# blood pressure.
df = df.drop(columns=['EverAlcohol', 'Smokedatleast100ciginlife'])

# Body and metabolic variables:
# Initially, BMI and WaistCircumference were both retained because they provide different adiposity information.
# DiabetesStatus will be included.
# Initially, HypertensionHistory and HypertensionMedication were both retained because both are relevant to BP.
# TotalCholesterol will be included because lipid levels may also be associated with blood pressure.


# 5.1. Define initial feature matrix and target
X = df.drop(columns=['SEQN', 'Systolic_BP_Mean'])
y = df['Systolic_BP_Mean']


# 5.2. Define numerical and categorical features for transformer
numerical_features = X.select_dtypes(include='number').columns
categorical_features = X.select_dtypes(include='object').columns


# 5.3. Define categories for proper references
categorical_categories = [
    ['Female', 'Male'],
    [
        'Non-Hispanic White',
        'Mexican American',
        'Other Hispanic',
        'Non-Hispanic Black',
        'Non-Hispanic Asian',
        'Other / Multi-Racial'
    ],
    ['US or Washington,DC', 'Others'],
    [
        'Never in the last year',
        '1-2 times/year',
        '2 times/week',
        '2-3 times/month',
        '3-4 times/week',
        '3-6 times/year',
        '7-11 times/year',
        'Every day',
        'Nearly every day',
        'Once/month',
        'Once/week',
        'Unknown'
    ],
    ['Not at all', 'Every day', 'Some days'],
    ['No', 'Yes'],
    ['No', 'Yes'],
    ['No', 'Borderline', 'Yes'],
    ['No', 'Yes'],
    ['No', 'Unknown', 'Yes']
]


# 5.4. Define categorical pipeline
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(
        categories=categorical_categories,
        drop='first'
    )),
])


# 5.5. Define numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])


# 5.6. Define a class for imputation of CigarettesPerDay

class SmokerMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = X[X > 0].median()
        return self

    def transform(self, X):
        return X.fillna(self.median_)

    def get_feature_names_out(self, input_features=None):
        return input_features


# 5.7. Define ordinary numerical columns without CigarettesPerDay
ordinary_numerical_columns = numerical_features.drop('CigarettesPerDay')


# 5.8. Create ColumnTransformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, ordinary_numerical_columns),
    ('cigarettes', SmokerMedianImputer(), ['CigarettesPerDay']),
    ('categorical', categorical_pipeline, categorical_features)
])


# 6. Baseline Multiple Linear Regression

# 6.1. Train/test split and model fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_processed, y_train)


# 6.2. Coefficient interpretation
feature_names = preprocessor.get_feature_names_out()

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
})


# 6.3. Reference categories
encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
print(encoder.categories_)  # first category is used as the reference because drop='first'


# 6.4. Baseline model evaluation
y_pred = model.predict(X_test_processed)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f} mmHg')  # 17.44
print(f'MAE:  {mae:.2f} mmHg')  # 13.01
print(f'R²:   {r2:.3f}')  # 0.254

# Baseline OLS model: RMSE = 17.44 mmHg, MAE = 13.01 mmHg, R² = 0.254.
# Predictive performance is modest, with the model explaining approximately 25% of the variance in systolic BP.


# 7. Multicollinearity Inspection

# 7.1. Calculate variance inflation factors
X_train_vif = sm.add_constant(X_train_processed)

vifs = []

for i in range(1, X_train_vif.shape[1]):
    vif = variance_inflation_factor(X_train_vif, i)
    vifs.append(vif)

feature_names_with_vifs = pd.DataFrame({
    'Features': feature_names,
    'VIFs': vifs
}).sort_values(by='VIFs', ascending=False)


# 7.2. Inspect redundant variables taking VIFs into account

# 7.2.1. HypertensionHistory vs. HypertensionMedication

# HypertensionHistory - VIF = 5.61 - clearer observed BP group differences
# HypertensionMedication - VIF = 5.37 - weaker observed BP group differences

pd.crosstab(
    df['HypertensionHistory'],
    df['HypertensionMedication']
)
# HypertensionMedication is strongly related to HypertensionHistory. Everyone without a hypertension history also has
# no hypertension medication, while medication status mainly differs among participants with a hypertension history.
# HypertensionHistory also shows a clearer relationship with systolic BP in the EDA. Therefore, HypertensionMedication
# can be considered for removal to reduce redundancy while retaining HypertensionHistory.


# 7.2.2. BMI vs. WaistCircumference

# Pearson correlation: r = 0.905
# VIF for WaistCircumference = 4.88; VIF for BMI = 4.58

# High and similar VIF values for BMI and WaistCircumference indicate that both variables have substantial linear
# relationships with the other predictors. Their Pearson correlation of 0.905 indicates a very strong pairwise linear
# relationship. In addition, WaistCircumference has a stronger simple linear association with systolic BP (r = 0.224)
# than BMI (r = 0.153). Therefore, WaistCircumference is the more defensible variable to retain in the initial model.


# 7.2.3. CurrentSmoking vs. CigarettesPerDay

print(df.groupby('CurrentSmoking')['CigarettesPerDay'].describe())

# CurrentSmoking and CigarettesPerDay are related, as daily smokers generally have higher cigarette consumption while
# people who do not smoke have 0 cigarettes per day. However, CigarettesPerDay also provides information about smoking
# intensity within the smoking groups. Therefore, both variables can be kept in the model.


# 8. Preprocessing for Reduced Model

X = df.drop(columns=[
    'SEQN',
    'Systolic_BP_Mean',
    'HypertensionMedication',
    'BMI(kg/m**2)'
])
y = df['Systolic_BP_Mean']

# Define numerical features for transformer
numerical_features = X.select_dtypes(include='number').columns

# Define categorical features for transformer
categorical_features = X.select_dtypes(include='object').columns

# Define categories for proper references
categorical_categories = [
    ['Female', 'Male'],
    [
        'Non-Hispanic White',
        'Mexican American',
        'Other Hispanic',
        'Non-Hispanic Black',
        'Non-Hispanic Asian',
        'Other / Multi-Racial'
    ],
    ['US or Washington,DC', 'Others'],
    [
        'Never in the last year',
        '1-2 times/year',
        '2 times/week',
        '2-3 times/month',
        '3-4 times/week',
        '3-6 times/year',
        '7-11 times/year',
        'Every day',
        'Nearly every day',
        'Once/month',
        'Once/week',
        'Unknown'
    ],
    ['Not at all', 'Every day', 'Some days'],
    ['No', 'Yes'],
    ['No', 'Yes'],
    ['No', 'Borderline', 'Yes'],
    ['No', 'Yes']
]

# Define categorical pipeline
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(
        categories=categorical_categories,
        drop='first'
    )),
])

# Define numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
])


# Define a class for imputation of CigarettesPerDay

class SmokerMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_ = X[X > 0].median()
        return self

    def transform(self, X):
        return X.fillna(self.median_)

    def get_feature_names_out(self, input_features=None):
        return input_features


# Define ordinary numerical columns without CigarettesPerDay
ordinary_numerical_columns = numerical_features.drop('CigarettesPerDay')

# Create ColumnTransformer
preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, ordinary_numerical_columns),
    ('cigarettes', SmokerMedianImputer(), ['CigarettesPerDay']),
    ('categorical', categorical_pipeline, categorical_features)
])


# 9. Reduced Multiple Linear Regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_processed, y_train)

# 9.1. Feature names
feature_names = preprocessor.get_feature_names_out()

# Dataframe containing feature names and coef
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_
})


# 9.2. Reduced model evaluation
y_pred = model.predict(X_test_processed)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f} mmHg')  # 17.50
print(f'MAE:  {mae:.2f} mmHg')  # 13.00
print(f'R²:   {r2:.3f}')  # 0.248


# Investigate coefficient change after removal of the potentially redundant variables
coef_df_after_removal = coef_df[
    coef_df['feature'].str.contains('WaistCircumference|HypertensionHistory')
]
# Removing BMI changed the WaistCircumference coefficient from −0.0249 to +0.1018.
# Removing HypertensionMedication changed the HypertensionHistory coefficient from 15.748 to 8.843.
# Model performance barely changed: R² 0.254 → 0.248.


# 10. Regression Assumptions

# 10.1. Linearity assumption

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel('Predicted Systolic BP')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Systolic BP')
plt.show()
# There is no obvious curved pattern in the residuals, although the spread of residuals appears to increase at higher
# predicted BP values.


# 10.2. Homoscedasticity assumption

# The Breusch–Pagan test
X_test_bp = sm.add_constant(X_test_processed)
bp_test = het_breuschpagan(residuals, X_test_bp)
print(f'Breusch-Pagan test results: \n{bp_test}')

# Breusch-Pagan test: LM statistic = 101.00, p-value = 2.42e-09.
# Since the p-value is much smaller than 0.05, we reject the null hypothesis of constant residual variance.
# This provides evidence of heteroscedasticity, which is also visible in the residual plot as a wider spread
# of residuals at higher predicted BP values.

# 10.3. Normality of residuals

# 10.3.1. Histogram
plt.hist(residuals, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

# Residuals are approximately normally distributed in the central region, but the tails show substantial deviation,
# particularly the upper tail.

# 10.3.2. Q-Q plot
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()

# The Q-Q plot agrees with the histogram, showing substantial deviation from the reference line in the tails,
# particularly the upper tail.

# 10.3.3. Shapiro-Wilk test
shapiro_test = shapiro(residuals)
print(f'Shapiro-Wilk test results: \n{shapiro_test}')
# The residuals are not normally distributed. Although the central part of the distribution is approximately
# bell-shaped, the residuals show positive skew and substantial deviation in the tails, particularly the upper tail.
# The Shapiro-Wilk test also provides strong evidence against normality (p = 3.00e-15).


# 10.4. Independence of residuals
# Independence is considered plausible because the modeling dataset contains one observation per participant and
# no repeated measurements or obvious clustering structure was introduced.


# 11. Results

# The final modeling dataset contained 5231 participants and the baseline model was fitted using 17 predictors, while
# the reduced model used 15 predictors. The baseline multiple linear regression model has the following evaluation
# metrics:
# RMSE: 17.44; MAE: 13.01; R²: 0.254
# and the reduced model has these evaluation metrics:
# RMSE: 17.50; MAE: 13.00; R²: 0.248
# Removing BMI caused a change in the coefficient of WaistCircumference from -0.025 to +0.102. Also, removal of
# HypertensionMedication changed the HypertensionHistory coefficient from 15.748 to 8.843.
# The baseline model showed slightly better performance, even though it is still modest for predictive purposes.
# Additionally, although there isn't an obvious curved pattern in the residuals, their spread shows a visible
# increase at higher predicted BP values. This result is further supported by the Breusch-Pagan test with an LM
# statistic = 101.00 and p-value = 2.42e-09. Also, residuals show an approximately normal distribution in the central
# region, but the tails show substantial deviation, especially the upper tail. Finally, independence of residuals was
# considered plausible because there was one observation per participant and no repeated measurements or obvious
# clustering structure.


# 12. Conclusion

# The multiple linear regression analysis provided modest predictive performance for systolic blood pressure.
# The baseline model performed slightly better than the reduced model, while removal of BMI and
# HypertensionMedication substantially changed the coefficients of WaistCircumference and HypertensionHistory,
# respectively. This demonstrates that multicollinearity can influence individual coefficient estimates without
# necessarily improving overall model performance.

# The regression assumptions were not fully satisfied, with evidence of heteroscedasticity and non-normal residuals.
# Independence of observations was considered plausible. As this analysis uses cross-sectional observational data,
# the observed relationships should not be interpreted as causal.


# 13. Limitations

# 1. Substantial or structural missingness limited the inclusion of some potentially relevant variables.
# 2. Several regression assumptions were not fully satisfied, including residual normality and homoscedasticity.
# 3. Causal relationships cannot be established because the dataset consists of cross-sectional observational data.
# 4. Multicollinearity was present among some predictors and affected coefficient estimates despite variable reduction.
# 5. The model's predictive performance was modest (R² around 0.25). Thus, other models can be tested for predictive
# purposes.
# 6. There was no external validation dataset, limiting assessment of how well the model generalizes to other
# populations.





