import os
import numpy as np
import pandas as pd
import random
import argparse

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from utils_ts import fun_df_cumsum, fun_cum_vec, get_PCT, get_FPR, get_TPR, fun_cum_tpr

def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Straggle Prediction on Live Data.')
  parser.add_argument('--data_path', type=str, help='Data path')
  parser.add_argument('--jobid', type=str, default="6343048076", help='Job ID') 
  parser.add_argument('--rs', type=int, default=42, help='Random state (default: 42)')
  parser.add_argument('--pt', type=float, default=0.2, help='Training set size (default: 0.2)')
  parser.add_argument('--tail', type=float, default=0.9, help='Latency threshold (default: 0.9)')
  parser.add_argument('--delta', type=float, default=0, help='Parameter for propensity score adjustment (default: 0)')
  parser.add_argument('--out', type=str, default='out', help='Output folder to save results (default: out)')

  args = parser.parse_args()
 
  path_ts   = args.data_path  
  jobid     = args.jobid
  delta     = args.delta  # Parameter to tune propensity score
  pt        = args.pt  ## Training set size
  tail      = args.tail # Latency threshold
  rs        = args.rs  # Random state
  out      = args.out  # Output folder to save results

  print("data_path: {}".format(path_ts))
  print("jobid:     {}".format(jobid))
  print("delta:     {}".format(delta))
  print("pt   :     {}".format(pt))
  print("tail:      {}".format(tail))
  print("rs   :     {}".format(rs))
  print("out :     {}".format(out))

  if not os.path.exists(out):
    os.makedirs(out)
    os.makedirs('{}/res_ts'.format(out))
    print(yes)

  path_ts_file = path_ts + jobid
  files_task = [f for f in listdir(path_ts_file) if isfile(join(path_ts_file, f))]
  job_rawsize = len(files_task)  ## Get number of tasks in a job

  task_colnames = ['ST','ET','JI', 'TI','MI', 'MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SP', 'AT', 'SCPU', 'EV', 'FL']
  task_fields = ['ST','ET','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']
  task_cols = ['Latency','MCU', 'CMU', 'AMU', 'UPC', 'TPC', 'MAXMU', 'MIO',
            'MDK', 'MAXCPU', 'MAXIO', 'CPI', 'MAI', 'SCPU', 'EV', 'FL']

  ## Get cumulative time series data
  list_task = [] 
  list_tp = []  ## list of total period
  list_task_compact = []  ## list of last row
  for i in range(job_rawsize):
    task = pd.read_csv('{}/{}/{}'.format(path_ts,jobid,i), header=None,
                       names=task_colnames, usecols=task_fields)
    task_new, tp_new = fun_df_cumsum(task)
    list_tp.append(tp_new)
    list_task_compact.append(task_new.iloc[-1].tolist())
    list_task.append(task_new)

  ## Construct new non-time series data
  np_task_compact = np.array(list_task_compact)
  df_task_compact = pd.DataFrame(np_task_compact, columns=task_fields)
  df_task_compact['Latency'] = pd.Series(np.asarray(list_tp), index=df_task_compact.index)
  df_sel = df_task_compact[task_cols]
  job = (df_sel-df_sel.min())/(df_sel.max()-df_sel.min())
  job = job.dropna(axis='columns') 
  job_raw = job.reset_index(drop=True)

  ## Normalize task at different time points using final row
  list_task_nn = []
  ts_size = 0  ## max task size in a job
  cn_train = [i for i in list(job) if i not in ['Latency']]  
  for i in range(len(list_task)):    
    task = list_task[i][cn_train]
    task = (task-job[cn_train].min())/(job[cn_train].max()-job[cn_train].min())
    if ts_size < task.shape[0]:
        ts_size = task.shape[0]
    list_task_nn.append(task)

  #####################################################################################
  ########## Now we have complete job data constructed from time series data ########## 
  #####################################################################################

  ## Split training and testing data
  latency = job_raw.Latency.values
  ## Parameter to tune propensity score
  lat_sort = np.sort(latency)

  print("# tail :  {}".format(tail))
  print("# delta:  {}".format(delta))

  cutoff = int(tail*latency.shape[0])
  alpha = lat_sort.tolist()[cutoff]
  print("# alpha:  {}".format(alpha))

  cutoff_pt = int(pt * latency.shape[0])
  alpha_pt = lat_sort.tolist()[cutoff_pt]
  train_idx_init = job.index[job['Latency'] < alpha].tolist()
  test_idx_init = job.index[job['Latency'] >= alpha].tolist()
  train_idx_removed = job.index[(job['Latency'] >= alpha_pt) & (job['Latency'] < alpha)].tolist()
  print("# true tail: {}".format(len(test_idx_init)))

  train_idx = list(set(train_idx_init) - set(train_idx_removed))
  test_idx = test_idx_init + train_idx_removed
  print("# removed: {}".format(len(train_idx_removed)))

  job =job_raw.copy()  ## this is VERY IMPORTANT!!!
  job_train = job.iloc[train_idx]
  job_test = job.iloc[test_idx]
  print("# train: {}".format(job_train.shape[0]))
  print("# test:  {}".format(job_test.shape[0]))

  X_train = job_train.to_numpy()[:,1:]
  Y_train = job_train.to_numpy()[:,0]
  X_test = job_test.to_numpy()[:,1:]
  Y_test = job_test.to_numpy()[:,0]

  job.loc[train_idx_init, 'Label'] = 0
  job.loc[test_idx_init, 'Label'] = 1
  y_test_true = job.loc[test_idx, 'Label'].values ## binary groundtruth for testing tasks
  y_stra_true = job.loc[test_idx_init, 'Latency'].values ## groundtruth for straggler

  ## Get latency bins, [90,95), [95, 99), [99+]
  cutoff95 = int(0.95 * latency.shape[0])
  alpha95 = lat_sort.tolist()[cutoff95]
  cutoff99 = int(0.99 * latency.shape[0])
  alpha99 = lat_sort.tolist()[cutoff99]
  test95_idx = job.index[(job['Latency'] >= alpha) & (job['Latency'] < alpha95)].tolist()
  test99_idx = job.index[(job['Latency'] >= alpha95) & (job['Latency'] < alpha99)].tolist()
  test99p_idx = job.index[(job['Latency'] >= alpha99)].tolist()
  BI = np.cumsum([len(test95_idx), len(test99_idx), len(test99p_idx)])
  print("# latency bins: {}".format(BI))

  ###################################################
  ########## Start time series experiments ########## 
  ###################################################

  ## Padding zero rows to unify task size
  list_task_norm = []
  test_idx_gap = [i for i in test_idx if i not in test_idx_init]
  list_task_nn_stra = [list_task_nn[i] for i in test_idx_init]  ## only straggler tasks
  list_task_nn_gap = [list_task_nn[i] for i in test_idx_gap] ## nonstragglers in testing
  list_task_nn_test = [list_task_nn[i] for i in test_idx]  ## for all test tasks

  ss_stra, ss_gap = [d.shape[0] for d in list_task_nn_stra], [d.shape[0] for d in list_task_nn_gap]
  ts_init_size = np.max(ss_stra)   ## max task size/time intervals for stragglers
          
  for dd in list_task_nn:
    if dd.shape[0] < ts_init_size:
      df2 =  pd.DataFrame(np.zeros([(ts_init_size-dd.shape[0]),dd.shape[1]]), columns=list(dd))
      list_task_norm.append(dd.append(df2, ignore_index=True))
    else:
      list_task_norm.append(dd)       
                 
  ## Only care about tasks that are stragglers
  list_task_norm_stra = [list_task_norm[i] for i in test_idx_init]
  list_task_norm_gap = [list_task_norm[i] for i in test_idx_gap]
  list_task_norm_test = [list_task_norm[i] for i in test_idx]

  ## Begin training
  X_train_up, X_test_up, Y_train_up, Y_test_up = X_train, X_test, Y_train, Y_test
  ## Train base models
  r_ridge = Ridge(alpha=0.1, copy_X=True,fit_intercept=False).fit(X_train_up, Y_train_up)
  r_svm = SVR(kernel='rbf').fit(X_train_up, Y_train_up)
  r_nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(100,50,10), random_state=rs).fit(X_train_up, Y_train_up)
  r_rf = RandomForestRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)
  r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)

  lt_stra, lt_gap = len(list_task_norm_stra), len(list_task_norm_gap)  ## straggler/non-straggler size
  list_task_norm_gap_down = list_task_norm_gap

  kl_stra_ridge, kl_stra_svm, kl_stra_nn, kl_stra_rf, kl_stra_gb, \
  kl_stra_ipw_ridge, kl_stra_ipw_svm, kl_stra_ipw_nn, kl_stra_ipw_rf, kl_stra_ipw_gb =\
  [], [], [], [], [], [], [], [], [], []

  kl_gap_ridge, kl_gap_svm, kl_gap_nn, kl_gap_rf, kl_gap_gb, \
  kl_gap_ipw_ridge, kl_gap_ipw_svm, kl_gap_ipw_nn, kl_gap_ipw_rf, kl_gap_ipw_gb =\
  [], [], [], [], [], [], [], [], [], []

  fl_gap_ridge, fl_gap_svm, fl_gap_nn, fl_gap_rf, fl_gap_gb, \
  fl_gap_ipw_ridge, fl_gap_ipw_svm, fl_gap_ipw_nn, fl_gap_ipw_rf, fl_gap_ipw_gb =\
  [], [], [], [], [], [], [], [], [], []

  for k in range(2,ts_init_size):
    p_stra_ridge, p_stra_svm, p_stra_nn, p_stra_rf, p_stra_gb, \
    p_stra_ipw_ridge, p_stra_ipw_svm, p_stra_ipw_nn, p_stra_ipw_rf, p_stra_ipw_gb = \
    np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra), \
    np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra),np.zeros(lt_stra)
        
    tn_stra = [i.iloc[k].values for i in list_task_norm_stra]
    np_tn_stra = np.asarray(tn_stra)
    np_tn_stra_nzidx = (np.where(np_tn_stra.any(axis=1))[0]).tolist()
    np_tn_stra_nz = np_tn_stra[~np.all(np_tn_stra == 0, axis=1)]
    
    tn_gap = [i.iloc[k].values for i in list_task_norm_gap_down]
    list_gap_idx = range(len(list_task_norm_gap_down))
    
    np_tn_gap = np.asarray(tn_gap)
      
    if len(np_tn_gap)>0 :
      np_tn_gap_zidx = (np.where(~np_tn_gap.any(axis=1))[0]).tolist()  ## indices of zero rows
      if len(np_tn_gap_zidx)>0:

        tn_gap_pre = [list_task_norm_gap[i].iloc[k-1].values for i in np_tn_gap_zidx]
        np_tn_gap_pre = np.asarray(tn_gap_pre)
        
        p_gap_ridge = r_ridge.predict(np_tn_gap_pre).tolist()
        p_gap_svm = r_svm.predict(np_tn_gap_pre).tolist()
        p_gap_nn = r_nn.predict(np_tn_gap_pre).tolist()
        p_gap_rf = r_rf.predict(np_tn_gap_pre).tolist()
        p_gap_gb = r_gb.predict(np_tn_gap_pre).tolist()
        
        kl_gap_ridge = kl_gap_ridge + p_gap_ridge
        kl_gap_svm = kl_gap_svm + p_gap_svm
        kl_gap_nn = kl_gap_nn + p_gap_nn
        kl_gap_rf = kl_gap_rf + p_gap_rf
        kl_gap_gb = kl_gap_gb + p_gap_gb            
        
        X = np.asarray(np.concatenate((X_train, np_tn_stra_nz, np_tn_gap_pre)))
        y = np.asarray([0] * X_train.shape[0] + [1] * np_tn_stra_nz.shape[0]+ [1] * np_tn_gap_pre.shape[0])
        clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
        ps = clf.predict_proba(X)
        ps1 = ps[(X_train.shape[0]+np_tn_stra_nz.shape[0]):,0] + delta    
        
        p_gap_ipw_ridge = [x/y for x, y in zip(p_gap_ridge, ps1.tolist())]
        p_gap_ipw_svm = [x/y for x, y in zip(p_gap_svm, ps1.tolist())]
        p_gap_ipw_nn = [x/y for x, y in zip(p_gap_nn, ps1.tolist())]
        p_gap_ipw_rf = [x/y for x, y in zip(p_gap_rf, ps1.tolist())]
        p_gap_ipw_gb = [x/y for x, y in zip(p_gap_gb, ps1.tolist())]
        
        kl_gap_ipw_ridge = kl_gap_ipw_ridge + p_gap_ipw_ridge 
        kl_gap_ipw_svm = kl_gap_ipw_svm + p_gap_ipw_svm
        kl_gap_ipw_nn = kl_gap_ipw_nn + p_gap_ipw_nn
        kl_gap_ipw_rf = kl_gap_ipw_rf + p_gap_ipw_rf
        kl_gap_ipw_gb = kl_gap_ipw_gb + p_gap_ipw_gb
        
        fl_gap_ridge.append(sum([1 for i in kl_gap_ridge if i>alpha])/len(kl_gap_ridge))
        fl_gap_svm.append(sum([1 for i in kl_gap_svm if i>alpha])/len(kl_gap_svm))
        fl_gap_nn.append(sum([1 for i in kl_gap_nn if i>alpha])/len(kl_gap_nn))
        fl_gap_rf.append(sum([1 for i in kl_gap_rf if i>alpha])/len(kl_gap_rf))
        fl_gap_gb.append(sum([1 for i in kl_gap_gb if i>alpha])/len(kl_gap_gb))
        fl_gap_ipw_ridge.append(sum([1 for i in kl_gap_ipw_ridge if i>alpha])/len(kl_gap_ipw_ridge))
        fl_gap_ipw_svm.append(sum([1 for i in kl_gap_ipw_svm if i>alpha])/len(kl_gap_ipw_svm))
        fl_gap_ipw_nn.append(sum([1 for i in kl_gap_ipw_nn if i>alpha])/len(kl_gap_ipw_nn))
        fl_gap_ipw_rf.append(sum([1 for i in kl_gap_ipw_rf if i>alpha])/len(kl_gap_ipw_rf))
        fl_gap_ipw_gb.append(sum([1 for i in kl_gap_ipw_gb if i>alpha])/len(kl_gap_ipw_gb))
    
        #print('# X_train_up before:   {}'.format(X_train_up.shape))       
        X_train_up = np.concatenate((X_train_up, X_test_up[np_tn_gap_zidx]))
        #print('# X_train_up after :   {}'.format(X_train_up.shape))
    
        #print('# Y_train_up before:   {}'.format(Y_train_up.shape))
        Y_train_up = np.concatenate((Y_train_up, Y_test_up[np_tn_gap_zidx]))
        #print('# Y_train_up after :   {}'.format(Y_train_up.shape))
    
        list_gap_idx = [i for i in list_gap_idx if i not in np_tn_gap_zidx]
        #print('list_gap_idx      :   {}'.format(list_gap_idx)) 
    
        list_task_norm_gap_down = [list_task_norm_gap_down[i] for i in list_gap_idx]
        #print('#list_task_norm_gap_down:{}'.format(len(list_task_norm_gap_down)))
    
        r_ridge = Ridge(alpha=0.1, copy_X=True,fit_intercept=False).fit(X_train_up, Y_train_up)
        r_svm = SVR(kernel='rbf').fit(X_train_up, Y_train_up)
        r_nn = MLPRegressor(alpha=1e-5, hidden_layer_sizes=(100,50,10), random_state=rs).fit(X_train_up, Y_train_up)
        r_rf = RandomForestRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)
        r_gb = GradientBoostingRegressor(n_estimators=100, random_state=rs).fit(X_train_up, Y_train_up)           
      
    p_stra_ridge[np_tn_stra_nzidx] = r_ridge.predict(np_tn_stra_nz)
    p_stra_svm[np_tn_stra_nzidx] = r_svm.predict(np_tn_stra_nz)
    p_stra_nn[np_tn_stra_nzidx] = r_nn.predict(np_tn_stra_nz)
    p_stra_rf[np_tn_stra_nzidx] = r_rf.predict(np_tn_stra_nz)    
    p_stra_gb[np_tn_stra_nzidx] = r_gb.predict(np_tn_stra_nz)    
    
    X = np.asarray(np.concatenate((X_train, np_tn_stra_nz)))
    y = np.asarray([0] * X_train.shape[0] + [1] * np_tn_stra_nz.shape[0])
    clf = LogisticRegression(random_state=rs, solver='lbfgs').fit(X, y)
    ps = clf.predict_proba(X)
    ps1 = ps[X_train.shape[0]:,0] + delta
    
    ## Prediction by IPW
    p_stra_ipw_ridge[np_tn_stra_nzidx] = p_stra_ridge[np_tn_stra_nzidx]/ ps1
    p_stra_ipw_svm[np_tn_stra_nzidx] = p_stra_svm[np_tn_stra_nzidx]/ ps1 
    p_stra_ipw_nn[np_tn_stra_nzidx] = p_stra_nn[np_tn_stra_nzidx]/ ps1 
    p_stra_ipw_rf[np_tn_stra_nzidx] = p_stra_rf[np_tn_stra_nzidx]/ ps1 
    p_stra_ipw_gb[np_tn_stra_nzidx] = p_stra_gb[np_tn_stra_nzidx]/ ps1 
    
    kl_stra_ridge.append(p_stra_ridge)    
    kl_stra_svm.append(p_stra_svm)
    kl_stra_nn.append(p_stra_nn)
    kl_stra_rf.append(p_stra_rf)
    kl_stra_gb.append(p_stra_gb)
    
    kl_stra_ipw_ridge.append(p_stra_ipw_ridge)
    kl_stra_ipw_svm.append(p_stra_ipw_svm)
    kl_stra_ipw_nn.append(p_stra_ipw_nn)
    kl_stra_ipw_rf.append(p_stra_ipw_rf)
    kl_stra_ipw_gb.append(p_stra_ipw_gb)

  #### Get percentile tail results
  PCT_ridge = get_PCT(kl_stra_ridge, y_stra_true, alpha, BI)
  PCT_svm = get_PCT(kl_stra_svm, y_stra_true, alpha, BI)
  PCT_nn = get_PCT(kl_stra_nn, y_stra_true, alpha, BI)
  PCT_rf = get_PCT(kl_stra_rf, y_stra_true, alpha, BI)
  PCT_gb = get_PCT(kl_stra_gb, y_stra_true, alpha, BI)

  PCT_ipw_ridge = get_PCT(kl_stra_ipw_ridge, y_stra_true, alpha, BI)
  PCT_ipw_svm = get_PCT(kl_stra_ipw_svm, y_stra_true, alpha, BI)
  PCT_ipw_nn = get_PCT(kl_stra_ipw_nn, y_stra_true, alpha, BI)
  PCT_ipw_rf = get_PCT(kl_stra_ipw_rf, y_stra_true, alpha, BI)
  PCT_ipw_gb = get_PCT(kl_stra_ipw_gb, y_stra_true, alpha, BI)

  ## Get percentile results dataframe
  np_pct = np.concatenate([np.asarray(PCT_ridge).reshape(1,-1),np.asarray(PCT_svm).reshape(1,-1),
                          np.asarray(PCT_nn).reshape(1,-1),np.asarray(PCT_rf).reshape(1,-1),
                          np.asarray(PCT_gb).reshape(1,-1),
                          np.asarray(PCT_ipw_ridge).reshape(1,-1),np.asarray(PCT_ipw_svm).reshape(1,-1),
                          np.asarray(PCT_ipw_nn).reshape(1,-1),np.asarray(PCT_ipw_rf).reshape(1,-1),
                          np.asarray(PCT_ipw_gb).reshape(1,-1)])
  df_pct = pd.DataFrame(np_pct, columns=['<95','<99','99+'], 
                        index=['ridge', 'svm', 'nn', 'rf', 'gb', 'ridge_ipw','svm_ipw','nn_ipw', 'rf_ipw','gb_ipw'])
  df_pct.to_csv('{}/res_ts/ptc/Job{}_pct.csv'.format(out,jobid))
  print("Tail percentile: ")
  print(df_pct)


  #### Get true positive rate
  TPR_ridge = get_TPR(kl_stra_ridge, alpha)
  TPR_svm = get_TPR(kl_stra_svm, alpha)
  TPR_nn = get_TPR(kl_stra_nn, alpha)
  TPR_rf = get_TPR(kl_stra_rf, alpha)
  TPR_gb = get_TPR(kl_stra_gb, alpha)
  TPR_ipw_ridge = get_TPR(kl_stra_ipw_ridge, alpha)
  TPR_ipw_svm = get_TPR(kl_stra_ipw_svm, alpha)
  TPR_ipw_nn = get_TPR(kl_stra_ipw_nn, alpha)
  TPR_ipw_rf = get_TPR(kl_stra_ipw_rf, alpha)
  TPR_ipw_gb = get_TPR(kl_stra_ipw_gb, alpha)

  ## Get false positive rate
  FPR_ridge = get_FPR(kl_gap_ridge, alpha)
  FPR_svm = get_FPR(kl_gap_svm, alpha)
  FPR_nn = get_FPR(kl_gap_nn, alpha)
  FPR_rf = get_FPR(kl_gap_rf, alpha)
  FPR_gb = get_FPR(kl_gap_gb, alpha)
  FPR_ipw_ridge = get_FPR(kl_gap_ipw_ridge, alpha)
  FPR_ipw_svm = get_FPR(kl_gap_ipw_svm, alpha)
  FPR_ipw_nn = get_FPR(kl_gap_ipw_nn, alpha)
  FPR_ipw_rf = get_FPR(kl_gap_ipw_rf, alpha)
  FPR_ipw_gb = get_FPR(kl_gap_ipw_gb, alpha)

  TPR_L = [TPR_ridge, TPR_svm, TPR_nn, TPR_rf, TPR_gb, TPR_ipw_ridge, TPR_ipw_svm, TPR_ipw_nn, TPR_ipw_rf, TPR_ipw_gb]
  FPR_L = [FPR_ridge, FPR_svm, FPR_nn, FPR_rf, FPR_gb, FPR_ipw_ridge, FPR_ipw_svm, FPR_ipw_nn, FPR_ipw_rf, FPR_ipw_gb]
  df_acc = pd.DataFrame(list(zip(TPR_L,FPR_L)), columns=['TPR', 'FPR'],
                       index=['ridge', 'svm', 'nn', 'rf', 'gb', 'ridge_ipw','svm_ipw','nn_ipw', 'rf_ipw','gb_ipw'])
  df_acc.to_csv('{}/res_ts/acc/Job{}_acc.csv'.format(out, jobid))
  print("Total TPR/FPR: ")
  print(df_acc)

  ## Get TPR CDF
  pr_tpr_ridge = fun_cum_tpr(kl_stra_ridge, alpha)
  pr_tpr_svm = fun_cum_tpr(kl_stra_svm, alpha)
  pr_tpr_nn = fun_cum_tpr(kl_stra_nn, alpha)
  pr_tpr_rf = fun_cum_tpr(kl_stra_rf, alpha)
  pr_tpr_gb = fun_cum_tpr(kl_stra_gb, alpha)
  pr_tpr_ipw_ridge = fun_cum_tpr(kl_stra_ipw_ridge, alpha)
  pr_tpr_ipw_svm = fun_cum_tpr(kl_stra_ipw_svm, alpha)
  pr_tpr_ipw_nn = fun_cum_tpr(kl_stra_ipw_nn, alpha)
  pr_tpr_ipw_rf = fun_cum_tpr(kl_stra_ipw_rf, alpha)
  pr_tpr_ipw_gb = fun_cum_tpr(kl_stra_ipw_gb, alpha)

  pr_tpr = (pr_tpr_ridge+pr_tpr_svm+pr_tpr_nn+pr_tpr_rf+pr_tpr_gb)/5
  pr_tpr_ipw = (pr_tpr_ipw_ridge+pr_tpr_ipw_svm+pr_tpr_ipw_nn+pr_tpr_ipw_rf+pr_tpr_ipw_gb)/5
  df_cdf_tpr = pd.DataFrame(list(zip(pr_tpr,pr_tpr_ipw)), columns=['Correlation', 'Causal'])
  df_cdf_tpr.to_csv('{}/res_ts/cdf_tpr/Job{}_cdf_tpr.csv'.format(out, jobid))
  print("TPR CDF: ")
  print(df_cdf_tpr)

  #### Get CDF FPR
  fl_fpr = np.array([fl_gap_ridge, fl_gap_svm, fl_gap_nn, fl_gap_rf, fl_gap_gb])
  pr_fpr =  np.average(fl_fpr, axis=0)*100
  fl_fpr_ipw = np.array([fl_gap_ipw_ridge, fl_gap_ipw_svm, fl_gap_ipw_nn, fl_gap_ipw_rf, fl_gap_ipw_gb])
  pr_fpr_ipw =  np.average(fl_fpr_ipw, axis=0)*100
  df_cdf_fpr = pd.DataFrame(list(zip(pr_fpr,pr_fpr_ipw)), columns=['Correlation', 'Causal'])
  df_cdf_fpr.to_csv('{}/res_ts/cdf_fpr/Job{}_cdf_fpr.csv'.format(out, jobid))
  print("FPR CDF: ")
  print(df_cdf_fpr)

if __name__ == '__main__':
  main()














