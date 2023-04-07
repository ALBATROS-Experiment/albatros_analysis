#/bin/bash
for dir in ~/albatros_data/uapishka_april_23/data_auto_cross/16807/* 
do
 echo "$dir"
 python3.8 quick_spectra.py -l -o ./plots/ "$dir"  
done

