#!/bin/bash

id="6343048076"
data_path='/Users/yiding2012/Desktop/sherlock/timeseriesspecsixtotal/'
data_path='/home/cc/timeseries5total/'
out="out5"

mkdir -p $out
mkdir -p "$out/res_ts"
mkdir -p "$out/res_ts/acc"
mkdir -p "$out/res_ts/cdf_tpr"
mkdir -p "$out/res_ts/cdf_fpr"
mkdir -p "$out/res_ts/ptc"
mkdir -p "$out/res_ts/time"

mkdir -p "$out/res_nts"
mkdir -p "$out/res_nts/acc"
mkdir -p "$out/res_nts/ptc"
mkdir -p "$out/res_nts/time"

mkdir -p "$out/res_apk"
mkdir -p "$out/res_apk/k3"
mkdir -p "$out/res_apk/k5"


mkdir -p "$out/res_adp"
mkdir -p "$out/res_seen"
mkdir -p "$out/res_tail"

mkdir -p "$out/res_agg"


#### Start run experiments

trap "exit" INT

cd ../timeseries5total

for d in * ; do
    echo "$d"
    cd ../sherlock
    python run_ts.py --data_path=$data_path --jobid=$d --delta=0.3 --pt=0.04 --tail=0.9 --rs=42 --out=$out
    cd ../timeseries5total
done



# python run_overall_timeseries.py --data_path=$data_path --jobid=$id --delta=0.3 --tail=0.9 --pt=0.2 --rs=42 --out=$out


# input="/Users/yiding2012/Desktop/causalDAGs/code/Straggler/jobidsfolder/jobidssub.txt"
# data_path="/Users/yiding2012/Desktop/sherlock/timeseriesspecsixtotal/"
# while IFS= read -r line
# do
#   echo "$line"
#   python run_overall_timeseries.py --jobid=$line --path=$data_path --delta=0.3 --pt=0.2 --tail=0.9 --rs=42 --out='out'
# done < "$input"





