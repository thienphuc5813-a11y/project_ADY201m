import numpy as np
import pandas as pd
from scipy import stats
from copy import deepcopy
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

# address outliers by using winsorization
def clip_outliers(df,col,threshold=1.5):

    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)

    IQR= Q3 - Q1

    if IQR==0:
        return df[col]

    higher_bound=Q3+threshold*IQR
    lower_bound=Q1-threshold*IQR

    return np.clip(df[col],a_min=lower_bound,a_max=higher_bound)


# cleaning data 
def cleaning_data(df,target_name='SalePrice',drop_threshold=0.6):
    # copy data for safety to avoid destroying our original data
    df_clean=df.copy()

    # drop columns including garbage values
    cols_to_drop = ['Id', 'Utilities', 'Street', 'Condition2', 'PoolQC']
    df_clean=df_clean.drop(columns=cols_to_drop,errors='ignore')

    X_clean=df_clean.drop(columns=target_name,errors='ignore')
    y_clean=df_clean[target_name]

    # drop columns including over 60% of null values
    null_percentages=X_clean.isnull().sum() / len(X_clean)
    drop_cols=null_percentages[null_percentages>drop_threshold].index
    X_clean=X_clean.drop(columns=drop_cols,errors='ignore')

    # becasue we're using boosting techniques so it is really sensitive to outliers,
    # we need to normalize our target based on its skewness to address this problem
    if y_clean.skew()>0.75: 
        y_clean=np.log1p(y_clean)

    # filter categorical and numeric columns
    numeric_cols=X_clean.select_dtypes(include=['int64','float64']).columns
    categorical_cols=X_clean.select_dtypes(include=['object']).columns

    # dealing with outliers for numeric cols
    for numeric_col in numeric_cols:
        X_clean[numeric_col]=clip_outliers(X_clean,numeric_col)

        skew_value=X_clean[numeric_col].skew()

        # in case skewness > 0.75, normalize this column by log
        # we use tree models to  predict our target so we dont need to normalize features
        # but we use knn technique to impute missing values so we must do this step
        if skew_value>0.75:
            X_clean[numeric_col]=np.log1p(X_clean[numeric_col])


    
    # filling null values for categorical features by most frequent value
    if len(categorical_cols)>0:
        cat_imputer=SimpleImputer(strategy='most_frequent')
        X_clean[categorical_cols]=cat_imputer.fit_transform(X_clean[categorical_cols])

    # filling null values for numeric features
    if len(numeric_cols)>0:
        # stadardize numeric features
        tmp_scaler=StandardScaler()
        num_scaled=tmp_scaler.fit_transform(X_clean[numeric_cols])
        # use knn to find 5 nearest neighbors
        knn=KNNImputer(n_neighbors=5)
        num_imputed=knn.fit_transform(num_scaled)

        X_clean[numeric_cols]=tmp_scaler.inverse_transform(num_imputed)

    # calculating p value to eliminate unnecessary columns (categorical type)
    good_cat_cols=[]

    for categorical_col in categorical_cols:
        groups=[group for name,group in y_clean.groupby(X_clean[categorical_col])]

        if len(groups)>1:
            f_stat,p_val=stats.f_oneway(*groups)
            # if p value <0.05, we keep this feature
            if p_val<0.05:
                good_cat_cols.append(categorical_col)

    bad_cat_cols=list(set(categorical_cols)-set(good_cat_cols))
    X_clean=X_clean.drop(columns=bad_cat_cols,errors='ignore')

    # building correlation matrix to eliminate numeric features having high correlation with each other
    X_temp=X_clean.copy()
    X_temp[target_name]=y_clean
    corr_matrix=X_temp.corr(numeric_only=True).abs()

    bad_numeric_cols=set()

    for i in range(len(numeric_cols)):
        for j in range(i+1,len(numeric_cols)):    
            feat1=numeric_cols[i]
            feat2=numeric_cols[j]
            # if correlation coefficient between 2 features > 0.8, we eliminate one of them based on the target
            if corr_matrix.loc[feat1,feat2]>0.8:
                if corr_matrix.loc[feat1,target_name]>corr_matrix.loc[feat2,target_name]:
                    bad_numeric_cols.add(feat2)
                else:
                    bad_numeric_cols.add(feat1)
    X_clean=X_clean.drop(columns=list(bad_numeric_cols),errors='ignore')

    # X_clean=pd.get_dummies(X_clean,drop_first=True)

    return X_clean,y_clean

