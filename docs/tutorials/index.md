This section provides hands-on examples and walkthroughs of some useful things our code can do.

## Identifying Batches

How to determine data presence for multiple antenna at once, and break them up into contiguous sections we call "batches".

## Unpacking Raw Data

How to move through long sections of data in chunks of a certain desired size. The Baseband File Iterator (BFI) object deals with holes in the data and moves between files seamlessly.

## Rechannelizing Data (re-PFB)

How to rechannelize baseband data to a desired frequency resolution.

## Running Satellite Detection

How to get coarse offsets and a list of high SNR pulses for a batch of data.

## Running Fine Timing

How to get the nanosecond level relative alignment between antenna timestreams.

## Unpacking Timing Solutions

How to access the data from Fine Timing, and how to apply it to data.