import numpy as np
import pandas as pd
import random
import time

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score

def fun_df_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Convert interval-wise time series data to cumulative time series data.
        Args:
            df: interval-wise time series data   
        return:
            df_new: cumulative time series data    
    '''
    fields = list(df)
    total_period = sum(df.ET-df.ST)  ## total trace period as latency
    task_weight = (df.ET-df.ST)/total_period
    task_weight = np.diag(task_weight.values)
    df_new = task_weight.dot(df)
    df_new = pd.DataFrame(data=df_new, columns=fields)
    return df_new.cumsum(), total_period

def get_mse(pred: np.array, truth: np.array) -> float:
    """Get MSE between prediction and groundtruth

    Args:
      pred: predicted array.
      truth: groundtruth array.

    Returns:
      return val: MSE.
    """
    return (np.square(pred - truth)).mean()

def get_score(y_true: np.array, y_pred: list, alpha:float, avg: str) -> list:
    """Get scores between prediction and groundtruth

    Args:    
      y_true: groundtruth binary array.
      y_pred: predicted numerical array.
      alpha: cutoff latency value.

    Returns:
      precision: precision
      recall: recall
      f1: f1
      auc: auc
    """
    y_b_pred = np.asarray([1 if i >= alpha else 0 for i in y_pred])
    score = precision_recall_fscore_support(y_true, y_b_pred, average=avg, labels=np.unique(y_b_pred))
    auc = roc_auc_score(y_true, y_b_pred)
    return [score[0], score[1], score[2], auc]

def get_recall(y_true_real: np.array, y_true: np.array, y_pred: list, alpha:float, bins: list) -> list:
    """Get recall between prediction and groundtruth

    Args:    
      y_true_real: groundtruth real value array.
      y_true: groundtruth binary array.
      y_pred: predicted numerical array.
      bins: bin size interval.

    Returns:
      recall: recall for [90,95), [95, 99), [99+]
    """
    y_pos_sort = np.argsort(y_true_real.reshape(-1))[(y_true_real.shape[0]-int(sum(y_true))):]
    y_b_pred = np.asarray([1 if i >= alpha else 0 for i in y_pred])
    
    s95 = recall_score(y_true[y_pos_sort[0:bins[0]].tolist()], y_b_pred[y_pos_sort[0:bins[0]].tolist()])
    s99 = recall_score(y_true[y_pos_sort[bins[0]:(bins[1])].tolist()], y_b_pred[y_pos_sort[bins[0]:bins[1]].tolist()])
    s99p = recall_score(y_true[y_pos_sort[bins[1]:(bins[2])].tolist()], y_b_pred[y_pos_sort[bins[1]:bins[2]].tolist()])
    return [s95, s99, s99p]


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    actual = np.array(actual).argsort()[::-1][:k].tolist()
    predicted = np.array(predicted).argsort()[::-1][:k].tolist()
    
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score*100 / min(len(actual), k)


def get_importance(labels: list, imp: list) -> pd.DataFrame:
    """Get importance dataframe

    Args:    
      labels: labels for all features.
      imp: list of importance scores.

    Returns:
      I: dataframe including importance per feature
    """
    I = pd.DataFrame(data={'Feature': labels, 'Importance': np.array(imp)})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I 


def get_agg_imp(imp: list) -> list:
    imp_mem = imp[1] + imp[2] + imp[5]
    imp_cpu = imp[0] + imp[8] + imp[12]
    imp_cache = imp[3] + imp[4]
    imp_io = imp[6] + imp[9] + imp[7]
    imp_uarch = imp[10] + imp[11]
    imp_ev = imp[13] + imp[14]
#     imp_fl = imp[14]
#     return [imp_mem, imp_cpu, imp_cache, imp_io, imp_uarch, imp_ev, imp_fl]
    return [imp_mem, imp_cpu, imp_cache, imp_io, imp_uarch, imp_ev]


## Function to get baseline results
def fun_imp(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, y_test_true: np.array, 
                alpha: float, niter: int, rs: int) -> np.array:
    """Get baseline and IPW prediction from training and testing data

    Args:    
      X_train: training data.
      Y_train: training numerical groundtruth.
      X_test: testing data.
      Y_test: testing numerical groundtruth.
      y_test_true: testing binary groundtruth.
      alpha: cutoff latency value.
      niter: number of iterations
      rs: random state.

    Returns:
      p_gb: baseline prediction with gradient boosting
      p_ipw_gb: ipw prediction with gradient boosting
    """

    r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train, Y_train)

    ### Baseline prediction 
    p_gb = r_gb.predict(X_test)
    
    ## Compute propensity scores
    
    X = np.asarray(np.concatenate((X_train, X_test)))
    y = np.asarray([0] * X_train.shape[0] + [1] * X_test.shape[0])
    clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)

    ps = clf.predict_proba(X)
    ps1 = ps[X_train.shape[0]:,0]

    ## Batch prediction by IPW
    p_ipw_gb = p_gb/ ps1    

    ## Get metrics scores
    avg = "weighted"

    s_gb = get_score(y_test_true, p_gb, alpha, avg)
    s_ipw_gb = get_score(y_test_true, p_ipw_gb, alpha, avg)

    imp_gb, imp_ipw_gb = [], []
    
    tm_start = time.time()
    for col in range(X_test.shape[1]):
      tmp_gb, tmp_ipw_gb = [], []
      for i in range(niter):
        save = X_test[:,col].copy() ## Create copy for original column
        X_test[:,col] = np.random.permutation(X_test[:,col]) ## Permute column
        ### New vanilla prediction 

        pm_gb = r_gb.predict(X_test)

        ## Recompute PS
        X = np.asarray(np.concatenate((X_train, X_test)))
        clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
        ps = clf.predict_proba(X)
        ps1 = ps[X_train.shape[0]:,0]

        X_test[:,col] = save
        ## New IPW prediction
        pm_ipw_gb = pm_gb/ ps1 

        ## Get scores
        sm_gb = get_score(y_test_true, pm_gb, alpha, avg)
        sm_ipw_gb = get_score(y_test_true, pm_ipw_gb, alpha, avg)

        tmp_gb.append(s_gb[2]-sm_gb[2])
        tmp_ipw_gb.append(s_ipw_gb[2]-sm_ipw_gb[2])

      tm = time.time() - tm_start

      imp_gb = imp_gb + [np.mean(tmp_gb)]
      imp_ipw_gb = imp_ipw_gb + [np.mean(tmp_ipw_gb)]

    ip_gb = get_agg_imp(imp_gb)
    ip_ipw_gb = get_agg_imp(imp_ipw_gb)

    return ip_gb, ip_ipw_gb, tm




## Function to get optimal results
def fun_imp_oracle(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, y_test_true: np.array, 
            alpha: float, niter: int, rs: int)-> np.array:
    """Get optimal prediction from training and testing data

    Args:    
      X_train: training data.
      Y_train: training numerical groundtruth.
      X_test: testing data.
      Y_test: testing numerical groundtruth.
      y_test_true: testing binary groundtruth.
      alpha: cutoff latency value.
      rs: random state.

    Returns:
      p_ipw_opt_gb: prediction with gradient boosting
    """
    X_all = np.asarray(np.concatenate((X_train, X_test)))
    Y_all = np.asarray(np.concatenate((Y_train, Y_test)))
    y_opt = np.asarray([0] * X_train.shape[0] + y_test_true.tolist())
    clf_opt = LogisticRegression(random_state=rs, solver='lbfgs').fit(X_all, y_opt)
    ## 1. PS
    ps_opt = clf_opt.predict_proba(X_all)
    ps1_opt = ps_opt[X_train.shape[0]:,0]

    r_opt_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_all, Y_all)

    ## Get prediction results
    p_opt_gb = r_opt_gb.predict(X_test)
    p_ipw_opt_gb = p_opt_gb/ps1_opt

    ## Get metrics scores
    avg = "weighted"

    s_opt_gb = get_score(y_test_true, p_opt_gb, alpha, avg)
    s_ipw_opt_gb = get_score(y_test_true, p_ipw_opt_gb, alpha, avg)

    imp_opt_gb, imp_ipw_opt_gb = [], []

    for col in range(X_test.shape[1]):
      tmp_opt_gb, tmp_ipw_opt_gb =  [], []
      for i in range(niter):
        save = X_test[:,col].copy() ## Create copy for original column
        X_test[:,col] = np.random.permutation(X_test[:,col]) ## Permute column

        ## Recompute PS
        X_all = np.asarray(np.concatenate((X_train, X_test)))
        clf_opt = LogisticRegression(random_state=rs, solver='lbfgs').fit(X_all, y_opt)
        ## 1. PS
        ps_opt = clf_opt.predict_proba(X_all)
        ps1_opt = ps_opt[X_train.shape[0]:,0]
        
        ## New OPT prediction
        pm_opt_gb = r_opt_gb.predict(X_test)
        pm_ipw_opt_gb = (r_opt_gb.predict(X_test)/ps1_opt).tolist()

        X_test[:,col] = save

        sm_opt_gb = get_score(y_test_true, pm_opt_gb, alpha, avg)
        sm_ipw_opt_gb = get_score(y_test_true, pm_ipw_opt_gb, alpha, avg)

        tmp_opt_gb.append(s_opt_gb[2]-sm_opt_gb[2])
        tmp_ipw_opt_gb.append(s_ipw_opt_gb[2]-sm_ipw_opt_gb[2])

      imp_opt_gb = imp_opt_gb + [np.mean(tmp_opt_gb)]
      imp_ipw_opt_gb = imp_ipw_opt_gb + [np.mean(tmp_ipw_opt_gb)]


    ip_opt_gb = get_agg_imp(imp_opt_gb)
    ip_ipw_opt_gb = get_agg_imp(imp_ipw_opt_gb)

    return ip_opt_gb, ip_ipw_opt_gb


## Function to get optimal results
def fun_imp_oracle_all(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, y_test_true: np.array, 
            alpha: float, niter: int, rs: int)-> np.array:
    """Get optimal prediction from training and testing data

    Args:    
      X_train: training data.
      Y_train: training numerical groundtruth.
      X_test: testing data.
      Y_test: testing numerical groundtruth.
      y_test_true: testing binary groundtruth.
      alpha: cutoff latency value.
      rs: random state.

    Returns:
      p_ipw_opt_ridge: prediction with ridge regression
      p_ipw_opt_svm: prediction with svm
      p_ipw_opt_nn: prediction with nn
      p_ipw_opt_rf: prediction with random forests
      p_ipw_opt_gb: prediction with gradient boosting
    """
    X_all = np.asarray(np.concatenate((X_train, X_test)))
    Y_all = np.asarray(np.concatenate((Y_train, Y_test)))
    y_opt = np.asarray([0] * X_train.shape[0] + y_test_true.tolist())
    clf_opt = LogisticRegression(random_state=rs, solver='lbfgs').fit(X_all, y_opt)
    ## 1. PS
    ps_opt = clf_opt.predict_proba(X_all)
    ps1_opt = ps_opt[X_train.shape[0]:,0]

    r_opt_ridge = Ridge(alpha=0.1, copy_X=True,fit_intercept=False).fit(X_all, Y_all)
    r_opt_svm = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1).fit(X_all, Y_all)
    r_opt_nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(100,50,10), random_state=rs).fit(X_train, Y_train)
    r_opt_rf = RandomForestRegressor(n_estimators=100, random_state=rs).fit(X_all, Y_all)
    r_opt_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_all, Y_all)

    ## Get prediction results

    p_opt_ridge = r_opt_ridge.predict(X_test)
    p_opt_svm = r_opt_svm.predict(X_test)
    p_opt_nn = r_opt_nn.predict(X_test)
    p_opt_rf = r_opt_rf.predict(X_test)
    p_opt_gb = r_opt_gb.predict(X_test)

    p_ipw_opt_ridge = p_opt_ridge/ps1_opt
    p_ipw_opt_svm = p_opt_svm/ps1_opt
    p_ipw_opt_nn = p_opt_nn/ps1_opt
    p_ipw_opt_rf = p_opt_rf/ps1_opt
    p_ipw_opt_gb = p_opt_gb/ps1_opt

    ## Get metrics scores
    avg = "weighted"

    s_opt_ridge = get_score(y_test_true, p_opt_ridge, alpha, avg)
    s_opt_svm = get_score(y_test_true, p_opt_svm, alpha, avg)
    s_opt_nn = get_score(y_test_true, p_opt_nn, alpha, avg)
    s_opt_rf = get_score(y_test_true, p_opt_rf, alpha, avg)
    s_opt_gb = get_score(y_test_true, p_opt_gb, alpha, avg)

    s_ipw_opt_ridge = get_score(y_test_true, p_ipw_opt_ridge, alpha, avg)
    s_ipw_opt_svm = get_score(y_test_true, p_ipw_opt_svm, alpha, avg)
    s_ipw_opt_nn = get_score(y_test_true, p_ipw_opt_nn, alpha, avg)
    s_ipw_opt_rf = get_score(y_test_true, p_ipw_opt_rf, alpha, avg)
    s_ipw_opt_gb = get_score(y_test_true, p_ipw_opt_gb, alpha, avg)

    # s_opt_ridge = get_mse(y_test_true, p_opt_ridge)
    # s_opt_svm = get_mse(y_test_true, p_opt_svm)
    # s_opt_nn = get_mse(y_test_true, p_opt_nn)
    # s_opt_rf = get_mse(y_test_true, p_opt_rf)
    # s_opt_gb = get_mse(y_test_true, p_opt_gb)

    # s_ipw_opt_ridge = get_mse(y_test_true, p_ipw_opt_ridge)
    # s_ipw_opt_svm = get_mse(y_test_true, p_ipw_opt_svm)
    # s_ipw_opt_nn = get_mse(y_test_true, p_ipw_opt_nn)
    # s_ipw_opt_rf = get_mse(y_test_true, p_ipw_opt_rf)
    # s_ipw_opt_gb = get_mse(y_test_true, p_ipw_opt_gb)

    imp_opt_ridge, imp_opt_svm, imp_opt_nn, imp_opt_rf, imp_opt_gb, \
    imp_ipw_opt_ridge, imp_ipw_opt_svm, imp_ipw_opt_nn, imp_ipw_opt_rf, imp_ipw_opt_gb = \
    [], [], [], [], [], [], [], [], [], []

    for col in range(X_test.shape[1]):
      tmp_opt_ridge, tmp_opt_svm, tmp_opt_nn, tmp_opt_rf, tmp_opt_gb, \
      tmp_ipw_opt_ridge, tmp_ipw_opt_svm, tmp_ipw_opt_nn, tmp_ipw_opt_rf, tmp_ipw_opt_gb = \
      [], [], [], [], [], [], [], [], [], []
      for i in range(niter):
        save = X_test[:,col].copy() ## Create copy for original column
        X_test[:,col] = np.random.permutation(X_test[:,col]) ## Permute column

        ## Recompute PS
        X_all = np.asarray(np.concatenate((X_train, X_test)))
        clf_opt = LogisticRegression(random_state=rs, solver='lbfgs').fit(X_all, y_opt)
        ## 1. PS
        ps_opt = clf_opt.predict_proba(X_all)
        ps1_opt = ps_opt[X_train.shape[0]:,0]
        
        ## New OPT prediction

        pm_opt_ridge = r_opt_ridge.predict(X_test)
        pm_opt_svm = r_opt_svm.predict(X_test)
        pm_opt_nn = r_opt_nn.predict(X_test)
        pm_opt_rf = r_opt_rf.predict(X_test)
        pm_opt_gb = r_opt_gb.predict(X_test)

        pm_ipw_opt_ridge = (r_opt_ridge.predict(X_test)/ps1_opt).tolist()
        pm_ipw_opt_svm = (r_opt_svm.predict(X_test)/ps1_opt).tolist()
        pm_ipw_opt_nn = (r_opt_nn.predict(X_test)/ps1_opt).tolist()
        pm_ipw_opt_rf = (r_opt_rf.predict(X_test)/ps1_opt).tolist()
        pm_ipw_opt_gb = (r_opt_gb.predict(X_test)/ps1_opt).tolist()

        X_test[:,col] = save

        sm_opt_ridge = get_score(y_test_true, pm_opt_ridge, alpha, avg)
        sm_opt_svm = get_score(y_test_true, pm_opt_svm, alpha, avg)
        sm_opt_nn = get_score(y_test_true, pm_opt_nn, alpha, avg)
        sm_opt_rf = get_score(y_test_true, pm_opt_rf, alpha, avg)
        sm_opt_gb = get_score(y_test_true, pm_opt_gb, alpha, avg)

        sm_ipw_opt_ridge = get_score(y_test_true, pm_ipw_opt_ridge, alpha, avg)
        sm_ipw_opt_svm = get_score(y_test_true, pm_ipw_opt_svm, alpha, avg)
        sm_ipw_opt_nn = get_score(y_test_true, pm_ipw_opt_nn, alpha, avg)
        sm_ipw_opt_rf = get_score(y_test_true, pm_ipw_opt_rf, alpha, avg)
        sm_ipw_opt_gb = get_score(y_test_true, pm_ipw_opt_gb, alpha, avg)

        # sm_opt_ridge = get_mse(y_test_true, pm_opt_ridge)
        # sm_opt_svm = get_mse(y_test_true, pm_opt_svm)
        # sm_opt_nn = get_mse(y_test_true, pm_opt_nn)
        # sm_opt_rf = get_mse(y_test_true, pm_opt_rf)
        # sm_opt_gb = get_mse(y_test_true, pm_opt_gb)

        # sm_ipw_opt_ridge = get_mse(y_test_true, pm_ipw_opt_ridge)
        # sm_ipw_opt_svm = get_mse(y_test_true, pm_ipw_opt_svm)
        # sm_ipw_opt_nn = get_mse(y_test_true, pm_ipw_opt_nn)
        # sm_ipw_opt_rf = get_mse(y_test_true, pm_ipw_opt_rf)
        # sm_ipw_opt_gb = get_mse(y_test_true, pm_ipw_opt_gb)

        tmp_opt_ridge.append(s_opt_ridge[2]-sm_opt_ridge[2])
        tmp_opt_svm.append(s_opt_svm[2]-sm_opt_svm[2])
        tmp_opt_nn.append(s_opt_nn[2]-sm_opt_nn[2])
        tmp_opt_rf.append(s_opt_rf[2]-sm_opt_rf[2])
        tmp_opt_gb.append(s_opt_gb[2]-sm_opt_gb[2])

        tmp_ipw_opt_ridge.append(s_ipw_opt_ridge[2]-sm_ipw_opt_ridge[2])
        tmp_ipw_opt_svm.append(s_ipw_opt_svm[2]-sm_ipw_opt_svm[2])
        tmp_ipw_opt_nn.append(s_ipw_opt_nn[2]-sm_ipw_opt_nn[2])
        tmp_ipw_opt_rf.append(s_ipw_opt_rf[2]-sm_ipw_opt_rf[2])
        tmp_ipw_opt_gb.append(s_ipw_opt_gb[2]-sm_ipw_opt_gb[2])

        # tmp_opt_ridge.append(s_opt_ridge-sm_opt_ridge)
        # tmp_opt_svm.append(s_opt_svm-sm_opt_svm)
        # tmp_opt_nn.append(s_opt_nn-sm_opt_nn)
        # tmp_opt_rf.append(s_opt_rf-sm_opt_rf)
        # tmp_opt_gb.append(s_opt_gb-sm_opt_gb)

        # tmp_ipw_opt_ridge.append(s_ipw_opt_ridge-sm_ipw_opt_ridge)
        # tmp_ipw_opt_svm.append(s_ipw_opt_svm-sm_ipw_opt_svm)
        # tmp_ipw_opt_nn.append(s_ipw_opt_nn-sm_ipw_opt_nn)
        # tmp_ipw_opt_rf.append(s_ipw_opt_rf-sm_ipw_opt_rf)
        # tmp_ipw_opt_gb.append(s_ipw_opt_gb-sm_ipw_opt_gb)

      imp_opt_ridge = imp_opt_ridge + [np.mean(tmp_opt_ridge)]
      imp_opt_svm = imp_opt_svm + [np.mean(tmp_opt_svm)]
      imp_opt_nn = imp_opt_nn + [np.mean(tmp_opt_nn)]
      imp_opt_rf = imp_opt_rf + [np.mean(tmp_opt_rf)]
      imp_opt_gb = imp_opt_gb + [np.mean(tmp_opt_gb)]

      imp_ipw_opt_ridge = imp_ipw_opt_ridge + [np.mean(tmp_ipw_opt_ridge)]
      imp_ipw_opt_svm = imp_ipw_opt_svm + [np.mean(tmp_ipw_opt_svm)]
      imp_ipw_opt_nn = imp_ipw_opt_nn + [np.mean(tmp_ipw_opt_nn)]
      imp_ipw_opt_rf = imp_ipw_opt_rf + [np.mean(tmp_ipw_opt_rf)]
      imp_ipw_opt_gb = imp_ipw_opt_gb + [np.mean(tmp_ipw_opt_gb)]

    ip_opt_ridge = get_agg_imp(imp_opt_ridge)
    ip_opt_svm = get_agg_imp(imp_opt_svm)
    ip_opt_nn = get_agg_imp(imp_opt_nn)
    ip_opt_rf = get_agg_imp(imp_opt_rf)
    ip_opt_gb = get_agg_imp(imp_opt_gb)
    ip_ipw_opt_ridge = get_agg_imp(imp_ipw_opt_ridge)
    ip_ipw_opt_svm = get_agg_imp(imp_ipw_opt_svm)
    ip_ipw_opt_nn = get_agg_imp(imp_ipw_opt_nn)
    ip_ipw_opt_rf = get_agg_imp(imp_ipw_opt_rf)
    ip_ipw_opt_gb = get_agg_imp(imp_ipw_opt_gb)

    return ip_opt_ridge, ip_opt_svm, ip_opt_nn, ip_opt_rf, ip_opt_gb, \
    ip_ipw_opt_ridge, ip_ipw_opt_svm, ip_ipw_opt_nn, ip_ipw_opt_rf, ip_ipw_opt_gb


## Function to get baseline results
def fun_imp_all(X_train: np.array, Y_train: np.array, X_test: np.array, Y_test: np.array, y_test_true: np.array, 
                alpha: float, niter: int, rs: int) -> np.array:
    """Get baseline and IPW prediction from training and testing data

    Args:    
      X_train: training data.
      Y_train: training numerical groundtruth.
      X_test: testing data.
      Y_test: testing numerical groundtruth.
      y_test_true: testing binary groundtruth.
      alpha: cutoff latency value.
      niter: number of iterations
      rs: random state.

    Returns:
      p_ridge: baseline prediction with ridge regression
      p_svm: baseline prediction with svm
      p_nn: baseline prediction with nn
      p_rf: baseline prediction with random forests
      p_gb: baseline prediction with gradient boosting
      p_ipw_ridge: ipw prediction with ridge regression
      p_ipw_svm: ipw prediction with svm
      p_ipw_nn: ipw prediction with nn
      p_ipw_rf: ipw prediction with random forests
      p_ipw_gb: ipw prediction with gradient boosting
    """
    r_ridge = Ridge(alpha=0.1, copy_X=True,fit_intercept=False).fit(X_train, Y_train)
    r_svm = SVR(kernel='rbf', gamma='scale', C=1.0, epsilon=0.1).fit(X_train, Y_train)
    r_nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(100,50,10), random_state=rs).fit(X_train, Y_train)
    r_rf = RandomForestRegressor(n_estimators=100,  random_state=rs).fit(X_train, Y_train)
    r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train, Y_train)

    ### Baseline prediction 
    p_ridge = r_ridge.predict(X_test)
    p_svm = r_svm.predict(X_test)
    p_nn = r_nn.predict(X_test)
    p_rf = r_rf.predict(X_test)
    p_gb = r_gb.predict(X_test)
    
    ## Compute propensity scores
    
    X = np.asarray(np.concatenate((X_train, X_test)))
    y = np.asarray([0] * X_train.shape[0] + [1] * X_test.shape[0])
    clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)

    ps = clf.predict_proba(X)
    ps1 = ps[X_train.shape[0]:,0]

    ## Batch prediction by IPW
    p_ipw_ridge = p_ridge/ ps1
    p_ipw_svm = p_svm/ ps1 
    p_ipw_nn = p_nn/ ps1
    p_ipw_rf = p_rf/ ps1 
    p_ipw_gb = p_gb/ ps1    

    ## Get metrics scores
    avg = "weighted"

    s_ridge = get_score(y_test_true, p_ridge, alpha, avg)
    s_svm = get_score(y_test_true, p_svm, alpha, avg)
    s_nn = get_score(y_test_true, p_nn, alpha, avg)
    s_rf = get_score(y_test_true, p_rf, alpha, avg)
    s_gb = get_score(y_test_true, p_gb, alpha, avg)

    s_ipw_ridge = get_score(y_test_true, p_ipw_ridge, alpha, avg)
    s_ipw_svm = get_score(y_test_true, p_ipw_svm, alpha, avg)
    s_ipw_nn = get_score(y_test_true, p_ipw_nn, alpha, avg)
    s_ipw_rf = get_score(y_test_true, p_ipw_rf, alpha, avg)
    s_ipw_gb = get_score(y_test_true, p_ipw_gb, alpha, avg)

    # s_ridge = get_mse(y_test_true, p_ridge)
    # s_svm = get_mse(y_test_true, p_svm)
    # s_nn = get_mse(y_test_true, p_nn)
    # s_rf = get_mse(y_test_true, p_rf)
    # s_gb = get_mse(y_test_true, p_gb)

    # s_ipw_ridge = get_mse(y_test_true, p_ipw_ridge)
    # s_ipw_svm = get_mse(y_test_true, p_ipw_svm)
    # s_ipw_nn = get_mse(y_test_true, p_ipw_nn)
    # s_ipw_rf = get_mse(y_test_true, p_ipw_rf)
    # s_ipw_gb = get_mse(y_test_true, p_ipw_gb)

    imp_ridge, imp_svm, imp_nn, imp_rf, imp_gb, imp_ipw_ridge, imp_ipw_svm, imp_ipw_nn, imp_ipw_rf, imp_ipw_gb =\
    [], [], [], [], [], [], [], [], [], []

    for col in range(X_test.shape[1]):
      tmp_ridge, tmp_svm, tmp_nn, tmp_rf, tmp_gb, tmp_ipw_ridge, tmp_ipw_svm, tmp_ipw_nn, tmp_ipw_rf, tmp_ipw_gb =\
      [], [], [], [], [], [], [], [], [], []
      for i in range(niter):
        save = X_test[:,col].copy() ## Create copy for original column
        X_test[:,col] = np.random.permutation(X_test[:,col]) ## Permute column
        ### New vanilla prediction 
        pm_ridge = r_ridge.predict(X_test)
        pm_svm = r_svm.predict(X_test)
        pm_nn = r_nn.predict(X_test)
        pm_rf = r_rf.predict(X_test)
        pm_gb = r_gb.predict(X_test)

        ## Recompute PS
        X = np.asarray(np.concatenate((X_train, X_test)))
        clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
        ps = clf.predict_proba(X)
        ps1 = ps[X_train.shape[0]:,0]

        X_test[:,col] = save
        ## New IPW prediction
        pm_ipw_ridge = pm_ridge/ ps1
        pm_ipw_svm = pm_svm/ ps1 
        pm_ipw_nn = pm_nn/ ps1
        pm_ipw_rf = pm_rf/ ps1 
        pm_ipw_gb = pm_gb/ ps1 

        ## Get scores
        sm_ridge = get_score(y_test_true, pm_ridge, alpha, avg)
        sm_svm = get_score(y_test_true, pm_svm, alpha, avg)
        sm_nn = get_score(y_test_true, pm_nn, alpha, avg)
        sm_rf = get_score(y_test_true, pm_rf, alpha, avg)
        sm_gb = get_score(y_test_true, pm_gb, alpha, avg)

        sm_ipw_ridge = get_score(y_test_true, pm_ipw_ridge, alpha, avg)
        sm_ipw_svm = get_score(y_test_true, pm_ipw_svm, alpha, avg)
        sm_ipw_nn = get_score(y_test_true, pm_ipw_nn, alpha, avg)
        sm_ipw_rf = get_score(y_test_true, pm_ipw_rf, alpha, avg)
        sm_ipw_gb = get_score(y_test_true, pm_ipw_gb, alpha, avg)

        # sm_ridge = get_mse(y_test_true, pm_ridge)
        # sm_svm = get_mse(y_test_true, pm_svm)
        # sm_nn = get_mse(y_test_true, pm_nn)
        # sm_rf = get_mse(y_test_true, pm_rf)
        # sm_gb = get_mse(y_test_true, pm_gb)

        # sm_ipw_ridge = get_mse(y_test_true, pm_ipw_ridge)
        # sm_ipw_svm = get_mse(y_test_true, pm_ipw_svm)
        # sm_ipw_nn = get_mse(y_test_true, pm_ipw_nn)
        # sm_ipw_rf = get_mse(y_test_true, pm_ipw_rf)
        # sm_ipw_gb = get_mse(y_test_true, pm_ipw_gb)

        tmp_ridge.append(s_ridge[2]-sm_ridge[2])
        tmp_svm.append(s_svm[2]-sm_svm[2])
        tmp_nn.append(s_nn[2]-sm_nn[2])
        tmp_rf.append(s_rf[2]-sm_rf[2])
        tmp_gb.append(s_gb[2]-sm_gb[2])

        tmp_ipw_ridge.append(s_ipw_ridge[2]-sm_ipw_ridge[2])
        tmp_ipw_svm.append(s_ipw_svm[2]-sm_ipw_svm[2])
        tmp_ipw_nn.append(s_ipw_nn[2]-sm_ipw_nn[2])
        tmp_ipw_rf.append(s_ipw_rf[2]-sm_ipw_rf[2])
        tmp_ipw_gb.append(s_ipw_gb[2]-sm_ipw_gb[2])

        # tmp_ridge.append(s_ridge-sm_ridge)
        # tmp_svm.append(s_svm-sm_svm)
        # tmp_nn.append(s_nn-sm_nn)
        # tmp_rf.append(s_rf-sm_rf)
        # tmp_gb.append(s_gb-sm_gb)

        # tmp_ipw_ridge.append(s_ipw_ridge-sm_ipw_ridge)
        # tmp_ipw_svm.append(s_ipw_svm-sm_ipw_svm)
        # tmp_ipw_nn.append(s_ipw_nn-sm_ipw_nn)
        # tmp_ipw_rf.append(s_ipw_rf-sm_ipw_rf)
        # tmp_ipw_gb.append(s_ipw_gb-sm_ipw_gb)

      imp_ridge = imp_ridge + [np.mean(tmp_ridge)]
      imp_svm = imp_svm + [np.mean(tmp_svm)]
      imp_nn = imp_nn + [np.mean(tmp_nn)]
      imp_rf = imp_rf + [np.mean(tmp_rf)]
      imp_gb = imp_gb + [np.mean(tmp_gb)]

      imp_ipw_ridge = imp_ipw_ridge + [np.mean(tmp_ipw_ridge)]
      imp_ipw_svm = imp_ipw_svm + [np.mean(tmp_ipw_svm)]
      imp_ipw_nn = imp_ipw_nn + [np.mean(tmp_ipw_nn)]
      imp_ipw_rf = imp_ipw_rf + [np.mean(tmp_ipw_rf)]
      imp_ipw_gb = imp_ipw_gb + [np.mean(tmp_ipw_gb)]

    ip_ridge = get_agg_imp(imp_ridge)
    ip_svm = get_agg_imp(imp_svm)
    ip_nn = get_agg_imp(imp_nn)
    ip_rf = get_agg_imp(imp_rf)
    ip_gb = get_agg_imp(imp_gb)
    ip_ipw_ridge = get_agg_imp(imp_ipw_ridge)
    ip_ipw_svm = get_agg_imp(imp_ipw_svm)
    ip_ipw_nn = get_agg_imp(imp_ipw_nn)
    ip_ipw_rf = get_agg_imp(imp_ipw_rf)
    ip_ipw_gb = get_agg_imp(imp_ipw_gb)
    return ip_ridge, ip_svm, ip_nn, ip_rf, ip_gb, \
          ip_ipw_ridge, ip_ipw_svm, ip_ipw_nn, ip_ipw_rf, ip_ipw_gb

