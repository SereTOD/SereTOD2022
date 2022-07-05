import logging, time, os

class _Config:
    def __init__(self):
        self.seed=6
        self.exp_name='temp'
        self.exp_path=''
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.mode='train'

        self.gpt_path='uer/gpt2-chinese-cluecorpussmall'
        self.data_path='Track2_data/processed_data.json'
        self.data_dir='Track2_data/'

        self.device=0
        self.batch_size=8
        self.gradient_accumulation_steps=4
        self.epoch_num=50
        self.eval_batch_size=32
        self.lr = 2e-5
        self.warmup_ratio=0.2
        self.pad_id=0


    def _init_logging_handler(self):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.mode=='train':
            file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(self.mode, self.exp_name, self.seed))
        else:
            file_handler = logging.FileHandler(os.path.join(self.gpt_path, 'eval_log.txt'))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()