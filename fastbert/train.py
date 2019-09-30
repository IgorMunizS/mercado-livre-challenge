import torch
import apex

from transformers.tokenization_bert import BertTokenizer

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging


PRETRAINED_PATH = '../../bert_model/model_out/'

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_PATH,
                                          do_lower_case=False)


databunch = BertDataBunch('../../../dados/', '../../../dados/',
                          tokenizer,
                          train_file='bert_train_pt.csv',
                          val_file='bert_val_pt.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=768,
                          max_seq_length=15,
                          multi_gpu=False,
                          multi_label=False,
                          model_type='bert')



OUTPUT_DIR = '../../bert_model/'
logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
                        databunch,
                        pretrained_path=PRETRAINED_PATH,
                        metrics=metrics,
                        device=device_cuda,
                        logger=logger,
                        output_dir=OUTPUT_DIR,
                        finetuned_wgts_path=None,
                        warmup_steps=10000,
                        multi_gpu=False,
                        is_fp16=True,
                        multi_label=False,
                        logging_steps=0)

for i in range(3):
    try:
        learner.fit(epochs=1,
                    lr=3e-4,
                    validate=True,  # Evaluate the model after each epoch
                    schedule_type="warmup_cosine",
                    optimizer_type="lamb")

        learner.save_model()
    except:
        print("Reiniciando treino")
    torch.cuda.empty_cache()

for i in range(3):
    learner.fit(epochs=1,
                lr=3e-5,
                validate=True,  # Evaluate the model after each epoch
                schedule_type="warmup_cosine",
                optimizer_type="lamb")

    learner.save_model()

    torch.cuda.empty_cache()