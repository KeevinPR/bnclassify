import rpy2.robjects as robjects
try:
    from rpy2.robjects import pandas2ri
except ImportError:
    import rpy2.robjects.pandas2ri as pandas2ri

from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import pandas as pd
import numpy as np

# Import R packages
bnclassify = importr('bnclassify')
base = importr('base')

def _pd_to_r(df):
    """Converts a pandas DataFrame to an R dataframe with factors."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_df = robjects.conversion.py2rpy(df)
    
    # Ensure factors for categorical data
    robjects.globalenv['tmp_df'] = r_df
    # We use stringsAsFactors = TRUE to ensure categoricals are treated as factors
    robjects.r('tmp_df <- as.data.frame(unclass(tmp_df), stringsAsFactors = TRUE)')
    r_df = robjects.globalenv['tmp_df']
    return r_df

def _r_to_pd_df(r_df):
    """Converts R DataFrame to Pandas DataFrame."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_df = robjects.conversion.rpy2py(r_df)
    return pd_df

# --- 1. Structure Learning Algorithms ---

def nb(class_var, df):
    """Naive Bayes."""
    r_df = _pd_to_r(df)
    model = bnclassify.nb(class_var, r_df)
    return model

def tan_cl(class_var, df, score='aic', root=None):
    """TAN Chow-Liu."""
    r_df = _pd_to_r(df)
    kwargs = {'score': score}
    if root:
        kwargs['root'] = root
    model = bnclassify.tan_cl(class_var, r_df, **kwargs)
    return model

def tan_hc(class_var, df, k=5, epsilon=0.01, smooth=0):
    """TAN Hill-Climbing."""
    r_df = _pd_to_r(df)
    model = bnclassify.tan_hc(class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

def kdb(class_var, df, k=5, kdbk=2, epsilon=0.01, smooth=0):
    """k-Dependence Bayesian Classifier."""
    r_df = _pd_to_r(df)
    model = bnclassify.kdb(class_var, r_df, k=k, kdbk=kdbk, epsilon=epsilon, smooth=smooth)
    return model

def aode(class_var, df):
    """Averaged One-Dependence Estimators."""
    r_df = _pd_to_r(df)
    model = bnclassify.aode(class_var, r_df)
    return model

def fssj(class_var, df, k=5, epsilon=0.01, smooth=0):
    """Forward Sequential Selection and Joining."""
    r_df = _pd_to_r(df)
    model = bnclassify.fssj(class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

def bsej(class_var, df, k=5, epsilon=0.01, smooth=0):
    """Backward Sequential Elimination and Joining."""
    r_df = _pd_to_r(df)
    model = bnclassify.bsej(class_var, r_df, k=k, epsilon=epsilon, smooth=smooth)
    return model

# --- 2. Parameter Learning ---

def lp(model, df, smooth=0.5, wanbia=None, awnb_trees=None, manb_prior=None):
    """Learn parameters (Bayes, MLE, WANBIA, AWNB, MANB)."""
    r_df = _pd_to_r(df)
    kwargs = {'dataset': r_df, 'smooth': smooth}
    if wanbia is not None:
        kwargs['wanbia'] = wanbia
    if awnb_trees is not None:
        kwargs['awnb_trees'] = awnb_trees
    if manb_prior is not None:
        kwargs['manb_prior'] = manb_prior
    
    updated_model = bnclassify.lp(model, **kwargs)
    return updated_model

# --- 3. Evaluate ---
import rpy2.robjects.packages as rpackages
stats = rpackages.importr('stats')

def cv(model, df, k=10, dag=True):
    """Cross-validation accuracy."""
    r_df = _pd_to_r(df)
    results = bnclassify.cv(model, dataset=r_df, k=k, dag=dag)
    # Return mean accuracy as float
    try:
         # Check if it's a list/vector and take mean or first element
         if hasattr(results, '__iter__'):
             return float(base.mean(results)[0])
         return float(results)
    except:
         return results

def aic(model, df):
    r_df = _pd_to_r(df)
    # AIC is generic from stats
    return float(stats.AIC(model, r_df)[0])

def bic(model, df):
    r_df = _pd_to_r(df)
    # BIC is generic from stats
    return float(stats.BIC(model, r_df)[0])

def log_lik(model, df):
    r_df = _pd_to_r(df)
    # logLik is generic from stats
    return float(stats.logLik(model, r_df)[0])

# --- 4. Prediction ---

def predict(model, df, prob=False):
    """Predicts classes or probabilities.
       Ensures factor levels in df match those in the model's CPTs to avoid R errors.
    """
    # 1. Get features and their levels from the model
    # We use features(model) to get variable names.
    # To get levels, we can iterate features and check params(model)[[feat]] dimnames?
    # Or simpler: The model in R usually has a 'dataset' attribute if we look deep, but not always.
    # The error "Levels in data set must match those in the CPTs" implies we need to conform the data.
    
    # Strategy: 
    #   a. Convert df to R dataframe usually.
    #   b. But we need to force levels from the model.
    #   c. 'bnclassify' doesn't seem to expose a 'levels' accessor easily.
    #      However, for discrete naive bayes, the CPTs have the levels.
    
    # Let's try to get levels using R helper code:
    robjects.globalenv['tmp_model'] = model
    r_code_get_levels = """
    function(model) {
       vars <- features(model)
       lvls <- list()
       p <- params(model)
       for (v in vars) {
          # Attempt to get levels from the CPT dimensions
          # For a node X, params(model)[[X]] is a table P(X | Parents). 
          # The dimension for X is usually the first one?
          # Actually, bnclassify stores params in a specific way.
          # simpler: attributes(p[[v]])$dimnames[[v]]
          lvls[[v]] <- dimnames(p[[v]])[[v]]
       }
       return(lvls)
    }
    """
    get_levels_func = robjects.r(r_code_get_levels)
    model_levels = get_levels_func(model)
    
    # 2. Iterate through df columns and apply Categorical with these levels
    # Note: model_levels is an R list (named).
    # We need to map it to python dict.
    py_levels = {}
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Convert R list to python list/dict?
        # R list with names -> dict
        names = model_levels.names
        for i, name in enumerate(names):
             py_levels[name] = list(model_levels[i])

    # 3. Adjust DataFrame
    df_adjusted = df.copy()
    for col, levels in py_levels.items():
        if col in df_adjusted.columns:
            # Force the column to be categorical with EXACT levels from model
            df_adjusted[col] = pd.Categorical(df_adjusted[col], categories=levels)
    
    # 4. Convert to R
    r_df = _pd_to_r(df_adjusted)
    
    # predict(object, newdata, prob = FALSE, ...)
    preds = bnclassify.predict_bnc_fit(model, r_df, prob=prob)
    
    if prob:
        # Returns a matrix of probabilities
        # Attempt to get column names (class labels)
        r_cols = base.colnames(preds)
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
             py_preds = robjects.conversion.rpy2py(preds)
        
        df_out = pd.DataFrame(py_preds)
        
        try:
             # Assign columns if available and length matches
             if r_cols is not None and len(r_cols) == df_out.shape[1]:
                 df_out.columns = list(r_cols)
        except Exception:
             pass
             
        return df_out
    else:
        # Returns a factor vector of classes
        with localconverter(robjects.default_converter + pandas2ri.converter):
            py_preds = robjects.conversion.rpy2py(preds)
        return np.array(py_preds)

# --- 5. Inspection (NEW WRAPPERS) ---

def params(model):
    """Returns CPTs as a list/dict of DataFrames."""
    try:
        # params(x) returns a list of CPTs
        r_params = bnclassify.params(model)
        # Convert each CPT to pandas
        py_params = {}
        names = r_params.names
        for i, name in enumerate(names):
             # Each element is a table/array
             cpt = r_params[i]
             # Convert table to DF
             robjects.globalenv['tmp_cpt'] = cpt
             robjects.r('tmp_cpt_df <- as.data.frame.table(tmp_cpt)')
             df_cpt = _r_to_pd_df(robjects.globalenv['tmp_cpt_df'])
             py_params[name] = df_cpt
        return py_params
    except Exception as e:
        return {'error': str(e)}

def nparams(model):
    """Number of free parameters."""
    return int(bnclassify.nparams(model)[0])

def narcs(model):
    """Number of arcs."""
    return int(bnclassify.narcs(model)[0])

def features(model):
    """List of features."""
    r_feats = bnclassify.features(model)
    return list(r_feats)

def class_var(model):
    """Class variable name."""
    r_cv = bnclassify.class_var(model)
    return str(r_cv[0])

def families(model):
    """Families of variables."""
    r_families = bnclassify.families(model)
    return list(r_families)

def modelstring(model):
    """Model string representation."""
    return str(bnclassify.modelstring(model)[0])

def edges(model):
    """Extracts edges for graph plotting."""
    try:
        # We can fallback to extracting from families or modelstring if explicit edge list func doesn't exist.
        # But 'bnclassify' doesn't have a direct 'edges' function in the CRAN doc?
        # It has 'families(x)'. Let's use that.
        # families(x) returns a list of families. Each family starts with node, followed by parents.
        r_fams = bnclassify.families(model)
        edge_list = []
        
        # r_fams is a named list. Names are child nodes. Values are character vectors (child, parent1, parent2...) or similar?
        # Let's inspect R doc: families(x) returns "Returns the family of each variable."
        
        # Alternative: modelstring(x) returns bnlearn format string "[A][B|A][C|A,B]..."
        # Parsing modelstring is robust.
        ms = modelstring(model)
        # Parse logic: [Child|Parent1:Parent2][Child2]...
        import re
        # Find all [content] blocks
        blocks = re.findall(r'\[(.*?)\]', ms)
        for block in blocks:
            if '|' in block:
                child, parents = block.split('|')
                parent_list = parents.split(':')
                for p in parent_list:
                    edge_list.append((p, child))
            else:
                # No parents
                pass
        return edge_list
    except:
        return []
