import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype, is_object_dtype
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def remove_high_corr_features(dataframe, threshold=0.95,
                              protected_cols=None):
    if protected_cols is None:
        protected_cols = [
            "Customer_return3Months",
            "Customer_return6Months",
        ]
    numeric_data = dataframe.select_dtypes(include=np.number)
    corr = numeric_data.corr().abs()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    upper_corr = corr.where(mask)
    dropped_cols = []
    for col in upper_corr.columns:
        if col in protected_cols:
            continue
        if (upper_corr[col] > threshold).any():
            dropped_cols.append(col)
    
    cleaned_df = dataframe.drop(columns=dropped_cols)
    return cleaned_df

def post_filtering(df):
    columns_to_drop = ['Area_Name','Customer_FraudCount','Discount','Summary','Price','DocsUploaded','Placed_ByAdmin','Selected_Expert_Fraud_HighCost','Selected_Expert_Fraud_Behavioral'
                   ,'Selected_Expert_Fraud_Damage','Selected_Expert_Fraud_Intermediation']
    columns_to_drop += [ col for col in df.columns if ('Unnamed' in col)]
    # delete constant columns
    un_rate = df.nunique() / len(df)
    columns_to_drop += list(un_rate[un_rate <= 1/ len(df)].index)
    
    # delete column with high missing rate
    missing_percentages = df.isnull().sum() / len(df) * 100
    missing_df = pd.DataFrame({"Column Name": missing_percentages.index, "Missing Percentage": missing_percentages.values})
    missing_df = missing_df.sort_values(by="Missing Percentage", ascending=True)
    High_missing = missing_df[missing_df['Missing Percentage'] > 50]['Column Name']
    
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    columns_to_drop += list(High_missing) + list(datetime_cols)
    
    # delete contract columns
    pattern = re.compile(r'Contract_\d+_(.*)')
    for col in df.columns:
        m = pattern.match(col)
        if m:
            columns_to_drop.append(col)
    
    df = df.drop(columns= columns_to_drop)        
    df = remove_high_corr_features(df)
    return df



def pre_filtering(df, filter_not_completed = True):
    df['IsCompleted'] = df[[f'Contract_{i}_State' for i in range(10)]].eq('WORKMAN_FINISHED').any(axis=1)
    # delete not completed orders
    if(filter_not_completed):
        df = df[df['IsCompleted'] == True]
        
    return df
    

        
def handle_time_columns(df):
    datetime_columns = [col for col in df.columns if 'Date' in col]
    for col in datetime_columns:
        if is_string_dtype(df[col]) or is_object_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        if pd.api.types.is_datetime64tz_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df



def handle_selected_contract(df):
    def extract_contract_features(df):
        pattern = re.compile(r'Contract_\d+_(.*)')
        suffixes = set()

        for col in df.columns:
            m = pattern.match(col)
            if m:
                suffixes.add(m.group(1))

        return suffixes
    def get_selected_contract(row):
        for i in range(10):
            col = f'Contract_{i}_State'
            if col in row and row[col] == 'WORKMAN_FINISHED':
                return i
        return np.nan
    df['Selected_Contract'] = df.apply(get_selected_contract, axis=1)

    suffixes = extract_contract_features(df)

    for suffix in suffixes:
        new_col = f'Selected_{suffix}'

        def extract_value(row, suffix=suffix):
            idx = row['Selected_Contract']
            if np.isnan(idx):
                return np.nan

            col_name = f'Contract_{int(idx)}_{suffix}'
            return row[col_name] if col_name in row else np.nan

        df[new_col] = df.apply(extract_value, axis=1)

    return df

def create_temporal_features(df):    
    def compute_time_to_selected_contract(row):
        selected = row['Selected_Contract']
        if pd.isna(selected):
            return None
        contract_col = f'Contract_{int(selected)}_CreatedDate'
        return (row[contract_col] - row['Creation_DateTime']).total_seconds() / 60
    
    df['Time_to_First_Contract'] = (df['Contract_0_CreatedDate'] - df['Creation_DateTime']).dt.total_seconds() / 60
    df['Time_to_Selected_Contract'] = df.apply(compute_time_to_selected_contract, axis=1)
    df['Time_to_Service'] =  (df['Start_DateTime'] - df['Creation_DateTime']).dt.total_seconds() / 60
    df['Customer_Account_Age'] = (df['Creation_DateTime'] - df['Customer_DateJoined']).dt.total_seconds() / 3600
    
    return df


def create_numerical_features(df):    
    df['Customer_Order_Rate'] = df['Customer_PreviousOrdersCount'] / (df['Customer_Account_Age'] + 1)
    df['Customer_Engagement'] = df['Customer_CharehPoints'] * df['Customer_PreviousOrdersCount'] 
    df['Low_Score_Rate'] = df['Customer_PreviousOrdersCount'] / (df['Customer_LowScoreCount'] + 1)
    df['Order_Hour'] = df['Creation_DateTime'].dt.hour
    df['Order_Month'] = df['Creation_DateTime'].dt.month
    df['Joined_Month'] = df['Customer_DateJoined'].dt.month
    df['Discount_Percentage'] = (df['Discount'] / (df['Price'] + 1)).clip(0, 1)
    
    return df


def handle_categorical_columns(df):
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        df[col] = df[col].astype(str)
        df[f'{col}_encoded'] = le.fit_transform(df[col])

    return df



if __name__ == '__main__':
    pipeline = Pipeline([
        ('pre_filtering', FunctionTransformer(pre_filtering)),
        ('handle_selected_contract', FunctionTransformer(handle_selected_contract)),
        ('handle_time_columns', FunctionTransformer(handle_time_columns)),
        ('create_temporal_features', FunctionTransformer(create_temporal_features)),
        ('create_numerical_features', FunctionTransformer(create_numerical_features)),
        ('post_filtering', FunctionTransformer(post_filtering)),
    ],verbose=True)
    
    file_path = "../Data/Achareh_Orders_Sampled_Tehran_900000.csv"
    chunk_size = 100_000
    total_rows = 900_000
    reader = pd.read_csv(file_path, chunksize=chunk_size)
    processed_chunks = []
    for chunk in tqdm(reader, total=9, desc="Processing chunks"):
        chunk = pipeline.fit_transform(chunk)   
        processed_chunks.append(chunk)



    df = pd.concat(processed_chunks, ignore_index=True)
    df.to_csv('./Processed_Achareh_Orders_Sampled_Tehran_900000.csv')
