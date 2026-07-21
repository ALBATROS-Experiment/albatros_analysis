# Pulse Times (Satellite Detection)

## Purpose

This is to test [get_satdet.py](../../scripts/get_satdet.md). 

The aim of this test is to demonstrate that the pipeline yields similar satellite detection results for different pulse start time and end times. The idea is that it is robust to slightly shifted regions, and that the underlying signal is properly identified and detected. 

## Method
This is done by examining the same pulse but will differently chunked sections, i.e. with silghtly shifted start times. This will ensure that detection is not position-dependent for high SNR regions.


## Expected Outcome
