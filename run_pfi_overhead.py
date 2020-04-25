import os
import numpy as np
import pandas as pd
import random
import time
import argparse

from os import listdir
from os.path import isfile, join

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor

from imblearn.over_sampling import SMOTE 

from utils_ts import fun_df_cumsum, fun_cum_vec, get_PCT, get_FPR, get_TPR, fun_cum_tpr
from utils_apk import fun_df_cumsum, apk, fun_imp, fun_imp_oracle, get_agg_imp

def main():

  #####################################################
  ########## Read data and simple processing ########## 
  #####################################################

  parser = argparse.ArgumentParser(description='Straggle Prediction on Live Data.')
  parser.add_argument('--data_path', type=str, help='Data path')
  parser.add_argument('--jobid', type=str, default="6343048076", help='Job ID') 
  parser.add_argument('--rs', type=int, default=42, help='Random state (default: 42)')
  parser.add_argument('--pt', type=float, default=0.04, help='Training set size (default: 0.2)')
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
    print("Result folder created")

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
  job.fillna(0, inplace=True)   ### THIS IS VERY IMPORTANT!!!!

  #####################################################################################
  ########## Now we have complete job data constructed from time series data ########## 
  #####################################################################################

  ## Split training and testing data
  latency = job.Latency.values
  ## Parameter to tune propensity score
  delta = 0
  lat_sort = np.sort(latency)
  tail = 0.9

  print("# tail :  {}".format(tail))
  print("# delta:  {}".format(delta))

  cutoff = int(tail*latency.shape[0])
  alpha = lat_sort.tolist()[cutoff]
  print("# alpha:  {}".format(alpha))

  pt = 0.2
  cutoff_pt = int(pt * latency.shape[0])
  alpha_pt = lat_sort.tolist()[cutoff_pt]
  train_idx_init = job.index[job['Latency'] < alpha].tolist()
  test_idx_init = job.index[job['Latency'] >= alpha].tolist()
  train_idx_removed = job.index[(job['Latency'] >= alpha_pt) & (job['Latency'] < alpha)].tolist()
  print("# true tail: {}".format(len(test_idx_init)))

  train_idx = list(set(train_idx_init) - set(train_idx_removed))
  test_idx = test_idx_init + train_idx_removed
  print("# removed: {}".format(len(train_idx_removed)))

  # job =job_raw.copy()  ## this is VERY IMPORTANT!!!
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
  y_test_true = job.loc[test_idx, 'Label'].values ## binary groundtruth for testing set

  ## randomize testing indices
  rand_test_idx = np.random.RandomState(seed=rs).permutation(X_test.shape[0]).tolist()
  X_test = X_test[rand_test_idx]
  Y_test = Y_test[rand_test_idx]
  y_test_true = y_test_true[rand_test_idx] 

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
  features = list(job.columns)[1:-1]
  labels = job.columns[1:-1]

  ###################################################
  ########## Start time series experiments ########## 
  ###################################################

  niter = 5
  ## Get importances for all models
  imp_gb, imp_ipw_gb, tm= fun_imp(X_train, Y_train, X_test, Y_test, y_test_true, alpha, niter, rs)
  print(tm)
  
if __name__ == '__main__':
  main()




