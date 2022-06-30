python main.py -mode train\
    -cfg batch_size=8\
    gradient_accumulation_steps=4\
    epoch_num=30 device=$1\
    exp_name=baseline