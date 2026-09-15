# Factors Associated with Systolic Blood Pressure

## Objective 
- Investigate factors associated with systolic blood pressure 
- Check regression assumptions 
- Predict systolic blood pressure using demographics, laboratory results, and questionnaires of participants

## Dataset
- NHANES 2017-2018

## Model Used
- Baseline multiple linear regression
- Reduced multiple linear regression

## Evaluation
- Root mean squared error (RMSE)
- Mean absolute error (MAE)
- R-Sqaured (R²)
- Residual diagnostics

## Results
| Model | RMSE | MAE | R² |
|---------|---------|---------|---------:|
| Multiple Linear Regression | 17.44 | 13.01 | 0.254 |
| Reduced Multiple Linear Regression| 17.50 | 13.00 | 0.248 | 

## Key Result
- Baseline and reduced models showed modest predictive performance, while multicollinearity substantially affected some coefficient estimates.

## Interpretation
- The multiple linear regression analysis provided modest predictive performance for systolic blood pressure.
- The baseline model performed slightly better than the reduced model. Removing redundant predictors substantially changed some coefficient estimates, demonstrating the influence of multicollinearity on coefficient estimates without necessarily improving overall model performance.
- The regression assumptions were not fully satisfied, with evidence of heteroscedasticity and non-normal residuals. Independence of observations was considered plausible. As this analysis uses cross-sectional observational data, the observed relationships should not be interpreted as causal.

## Limitations
- Substantial or structural missingness limited the inclusion of some potentially relevant variables.
- Several regression assumptions were not fully satisfied, including residual normality and homoscedasticity.
- Causal relationships cannot be established because the dataset consists of cross-sectional observational data.
- Multicollinearity was present among some predictors and affected coefficient estimates despite variable reduction.
- The model's predictive performance was modest (R² around 0.25). Thus, other models can be tested for predictive purposes.
- There was no external validation dataset, limiting assessment of how well the model generalizes to other populations.