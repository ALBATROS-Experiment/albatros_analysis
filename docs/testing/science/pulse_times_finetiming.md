# Pulse Times (Fine Timing)

## Purpose

This is to test both [get_batch_discrepancies.py](../../scripts/get_batch_discrepancies.md) and [get_batch_finetiming.py](../../scripts/get_batch_finetiming.md). 

The aim of this test is to demonstrate that the fine timing section yields similar results for different pulse start time and end times. The idea is that it is robust to slightly shifted regions, and that the underlying signal is properly identified and fitted for. 

## Method
This is done by generating three satellite pulses on the same data, but with different UTC start and end times. Using a previously determiend UTC map, for the different pulses we examine:
    - fit to UTC time (should be similar fit)
    - cutting to low phase noise region (should be same region)
    - phase unwrapping plots (same signal should look similar)
    - tau values
Both scripts generate extensive debugplots, so it may also be useful to go into the directories to check they all look alright.

## Expected Outcome
You should get the same net offset, but the delay structure will likely look different since they start at different times. 