# Z2H

This is an anonymous repository holding the source code and mice data for ICDM 2024 paper DM761.

This repository is intended for review purposes only. Please do not use it for any other purposes, or redistribute it in any form. Thank you!

## Downloading the mice data
To download the mice data, follow this [link](https://drive.google.com/file/d/1Lzo7PLL6PrHqdwgvzbmkK4TpBiIyDjoQ/view?usp=sharing).

## Running the code
Suppose you have unzipped the mice data into the same directory as the code, follow the steps below to run it.

### 1. Pre-compute the Nearest Neighbor (NN) distance matrices
As a warmup routine, run the following to pre-compute the NN distances, which will be used in both Modules M1 and M2.

```
cd $code_dir
python ./Warmup_calc_dist_mat.py $data_id
```

Here, `$data_id` can be one of `MO1`, `MO2`, or `MO3`.

### 2. Execute Module M1 (Knowledge-guided Active Initialization)
To run Module M1, use the following.

```
python ./M1_active_initialization.py $data_id
```

### 3. Execute Module M2 (Actively Enhanced PU Transduction)
To run Module M2 to get a fully labeled training set, use the following.

```
python ./M2_active_pu_transduction.py $data_id
```

### 4. Execute Module M3 (Knowledge-guided Robust Deep Model Training)
To run Module M3, first use the following to apply the Knowledge-guided Training Example Filtering (KTEF) routine.

```
python ./M3_filter_examples.py $data_id
```

This will lead to the filtered dataset stored at `./result/KTEF_$data_id` by default. You can use it to feed it into either WNG-TS-1DCNN or iEDeal with the original authors' implementations.
