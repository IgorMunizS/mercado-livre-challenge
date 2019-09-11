from finetune.base_models import BERT, BERTLarge, GPT2, GPT2Medium, GPT2Large, TextCNN, TCN, RoBERTa, DistilBERT
from finetune import Classifier



def get_bert_model(batch_size,maxlen,dsize,save_path):
    model = Classifier(base_model=BERT, batch_size=batch_size, n_epochs=2, max_length=maxlen,
                       lr_schedule='warmup_linear', dataset_size=dsize, val_size=0.1,
                       autosave_path=save_path, class_weights='sqrt')


    return model
