@echo off
setlocal EnableDelayedExpansion

set fname[1]="epsmax_0.1_n100000.csv"
set fname[2]="epsmax_0.5_n100000.csv"
set fname[3]="epsmax_1.0_n100000.csv"

set epsmax[1]=0.1
set epsmax[2]=0.5
set epsmax[3]=1.0

for /l %%x in (1, 1, 3) do (
   python state_err_cmd.py --n_runs 100000 --eps_max !epsmax[%%x]! --file_name !fname[%%x]!
)
