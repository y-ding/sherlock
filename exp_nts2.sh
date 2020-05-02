#!/bin/bash

id="6343048076"
data_path='/Users/yiding2012/Desktop/sherlock/timeseriesspecsixtotal/'
data_path='/home/cc/timeseries2total/'
out="out2"

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

cd ../timeseries2total

for d in * ; do
    echo "$d"
    cd ../sherlock
    python run_nts.py --data_path=$data_path --jobid=$d --delta=0 --pt=0.2 --tail=0.9 --rs=42 --out=$out
    cd ../timeseries2total
done




