# This is repo to finetune smolVLA on AD dataset

## Step 1 : Conda env
```
conda activate lerobot
```

## Step 2 : Copy data 
Copy LTA_Pool/ folder from spectre location: "/home/AutoDP/rohit.pawar/lerobot/examples/tutorial/smolvla/LTA_Pool" into cloned "lerobot/examples/tutorial/smolvla/" path. 

## Step 3 : Training Scritps 
Train file 1 : Random weighst with Xavier weights are initialised for action decoder
```
python train_random.py 
```
Train file 2 : Both captions1 and captions2 are going for training with normalised actions from [-1,1]
```
python train_both_prompt.py 
```

Train file 3 : Vanilla script for training with un-normlaised action chunk and single captions1 finetunning
```
python training_loop_smolVLA.py 
```

## Step 4 : Inference step
Add checkpoints path and run: 
```
python inference.py
```
