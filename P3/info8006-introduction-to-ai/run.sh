#!/bin/bash
echo "" > log.txt
for w in 1 3 5; do
#for w in 1 2 3; do
  for p in 0.0 0.2 0.4 0.6 0.8 1; do
  #for p in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1; do
    #for ((i=0;i<2;i++));do

      echo "w = $w, p = $p, i = $i" >> log.txt
      python run.py --layout observer.lay --bsagentfile beliefstateagent.py --w ${w} --p ${p} --ghostagent leftrandy --nghosts 1 >> log.txt 2>> log.txt

  done
done
