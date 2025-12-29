import rpy2.robjects as robjects
try:
    from rpy2.robjects import pandas2ri
except ImportError:
    import rpy2.robjects.pandas2ri as pandas2ri

from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Import R packages
importr('bnclassify')

def _pd_to_r(df):
    """Converts a pandas DataFrame to an R dataframe with factors."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)
    # Ensure factors for categorical data
    robjects.globalenv['tmp_df'] = r_df
    robjects.r('tmp_df <- as.data.frame(unclass(tmp_df), stringsAsFactors = TRUE)')
    r_df = robjects.globalenv['tmp_df']
    return r_df

def nb(class_var, df):
    r_df = _pd_to_r(df)
    model = robjects.r['nb'](class_var, r_df)
    return model

def tan_cl(class_var, df, score='aic', root=None):
    r_df = _pd_to_r(df)
    kwargs = {'score': score}
    if root:
        kwargs['root'] = root
    model = robjects.r['tan_cl'](class_var, r_df, **kwargs)
    return model

def tan_hc(class_var, df, k=5, epsilon=0.01, smooth=0):
    """
    Hill-climbing tree augmented naive Bayes (TAN-HC).
    Uses cross-validated accuracy to guide the search.
    
    Args:
        class_var: Name of the class variable
        df: Dataset
        k: Number of folds for cross-validation (default=5)
        epsilon: Minimum improvement threshold (default=0.01)
        smooth: Smoothing parameter for Bayesian estimation (default=0)
    """
    r_df = _pd_to_r(df)
    model = robjects.r['tan_hc'](class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

def kdb(class_var, df, k=5, kdbk=2, epsilon=0.01, smooth=0):
    """
    k-Dependence Bayesian Classifier.
    
    Args:
        class_var: Name of the class variable
        df: Dataset
        k: Number of folds for cross-validation (default=5)
        kdbk: Maximum number of feature parents per feature (default=2)
        epsilon: Minimum improvement threshold (default=0.01)
        smooth: Smoothing parameter for Bayesian estimation (default=0)
    """
    r_df = _pd_to_r(df)
    model = robjects.r['kdb'](class_var, r_df, k=k, kdbk=kdbk, epsilon=epsilon, smooth=smooth)
    return model

def aode(class_var, df):
    r_df = _pd_to_r(df)
    model = robjects.r['aode'](class_var, r_df)
    return model

def fssj(class_var, df, k=5, epsilon=0.01, smooth=0):
    r_df = _pd_to_r(df)
    model = robjects.r['fssj'](class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

def bsej(class_var, df, k=5, epsilon=0.01, smooth=0):
    r_df = _pd_to_r(df)
    model = robjects.r['bsej'](class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

def lp(model, df, smooth=0.5, wanbia=None, awnb_trees=None, manb_prior=None):
    r_df = _pd_to_r(df)
    kwargs = {'dataset': r_df, 'smooth': smooth}
    if wanbia is not None:
        kwargs['wanbia'] = wanbia
    if awnb_trees is not None:
        kwargs['awnb_trees'] = awnb_trees
    if manb_prior is not None:
        kwargs['manb_prior'] = manb_prior
    
    # lp in bnclassify can take a bnc_dag or bnc_bn
    updated_model = robjects.r['lp'](model, **kwargs)
    return updated_model

def cv(model, df, k=10, dag=True):
    r_df = _pd_to_r(df)
    # cv(models, dataset, k, dag = TRUE, ...)
    results = robjects.r['cv'](model, dataset=r_df, k=k, dag=dag)
    # Results is usually a number (accuracy) or a list
    return results

def aic(model, df):
    r_df = _pd_to_r(df)
    result = robjects.r['AIC'](model, r_df)
    # Extract scalar value from R object
    try:
        return float(result[0]) if hasattr(result, '__getitem__') else float(result)
    except (TypeError, IndexError):
        return float(result)

def bic(model, df):
    r_df = _pd_to_r(df)
    result = robjects.r['BIC'](model, r_df)
    # Extract scalar value from R object
    try:
        return float(result[0]) if hasattr(result, '__getitem__') else float(result)
    except (TypeError, IndexError):
        return float(result)

def log_lik(model, df):
    r_df = _pd_to_r(df)
    result = robjects.r['logLik'](model, r_df)
    # Extract scalar value from R object
    try:
        return float(result[0]) if hasattr(result, '__getitem__') else float(result)
    except (TypeError, IndexError):
        return float(result)

def predict(model, df, prob=False):
    r_df = _pd_to_r(df)
    # predict(object, newdata, prob = FALSE, ...)
    preds = robjects.r['predict'](model, r_df, prob=prob)
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        py_preds = robjects.conversion.rpy2py(preds)
    
    return py_preds
