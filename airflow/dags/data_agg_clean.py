import numpy as np
import pandas as pd

def data_agg_clean_full(record):
    # Convert single message dict to DataFrame
    df = pd.DataFrame([record])

    df['is_train'] = None

    credit_bureau_cols = [col for col in df.columns if 'CREDIT_BUREAU' in col]
    df['FLAG_CREDIT_BUREAU_MISSING'] = df[credit_bureau_cols].isnull().any(axis=1).astype(int)

    df['FLAG_EXT_SOURCE_3_MISSING'] = df['EXT_SOURCE_3'].isnull().astype(int)
    df['FLAG_NO_CAR'] = df['OWN_CAR_AGE'].isnull().astype(int)
    df['FLAG_EXT_SOURCE_1_MISSING'] = df['EXT_SOURCE_1'].isnull().astype(int)

    high_missing_cols = [
        'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
        'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE',
        'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI',
        'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
        'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
    ]
    num_cols = df[high_missing_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df[high_missing_cols].select_dtypes(include=['object']).columns.tolist()
    building_cols = [col for col in num_cols + cat_cols if 'AVG' in col or 'MODE' in col or 'MEDI' in col]
    df['FLAG_BUILDING_INFO_MISSING'] = df[building_cols].isnull().any(axis=1).astype(int)

    df['FLAG_UNEMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365

    conditions = [
        df['DAYS_EMPLOYED'] == 365243,
        df['DAYS_EMPLOYED'] < -3650,
        df['DAYS_EMPLOYED'] < -730,
        df['DAYS_EMPLOYED'] < 0
    ]
    labels = ['Unemployed', 'Long_Term', 'Mid_Term', 'Short_Term']
    df['EMPLOYMENT_STATUS'] = np.select(conditions, labels, default='Unknown')

    df['REVOLVING_GOODS_OVER_CREDIT'] = (
        (df['NAME_CONTRACT_TYPE'] == 'Revolving loans') & 
        (df['AMT_GOODS_PRICE'] > df['AMT_CREDIT'])
    ).astype(int)

    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['CTI_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['HIGH_CTI'] = df['CTI_RATIO'] > 20
    df['HIGH_CTI_RISK'] = (df['CTI_RATIO'] > 20).astype(int)
    df['ATI_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['HIGH_ATI'] = df['ATI_RATIO'] > 0.5
    df['HIGH_ATI_RISK'] = (df['ATI_RATIO'] > 0.5).astype(int)
    df['GTI_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    df['HIGH_GTI'] = df['GTI_RATIO'] > 25
    df['HIGH_GTI_RISK'] = (df['GTI_RATIO'] > 25).astype(int)
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365.25

    # Return as a dict (first row only since it's a single record)
    return df.iloc[0].to_dict()
