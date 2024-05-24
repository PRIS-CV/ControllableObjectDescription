import transformers
import torch
import numpy as np
import random

from sklearn.metrics import average_precision_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SingleClassMetric(object):
    
    def __init__(self, class_name, pred, gt_label):
        """This class computes all metrics for a single attribute.

        Args:
        - pred: np.array of shape [n_instance] -> binary prediction.
        - gt_label: np.array of shape [n_instance] -> groundtruth binary label.
        """
        self.class_name = class_name
        if pred is None or gt_label is None:
            self.true_pos = 0
            self.false_pos = 0
            self.true_neg = 0
            self.false_neg = 0
            self.n_pos = 0
            self.n_neg = 0
            self.ap = -1
            return

        self.true_pos = ((gt_label == 1) & (pred == 1)).sum()
        self.false_pos = ((gt_label == 0) & (pred == 1)).sum()
        self.true_neg = ((gt_label == 0) & (pred == 0)).sum()
        self.false_neg = ((gt_label == 1) & (pred == 0)).sum()

        # Number of groundtruth positives & negatives.
        self.n_pos = self.true_pos + self.false_neg
        self.n_neg = self.false_pos + self.true_neg
        
        self.gt_label = gt_label
        self.pred = pred
        
    def get_ap(self):
        return average_precision_score(self.gt_label, self.pred)

    def get_recall(self):
        """Computes recall.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 0

        if self.n_pos > 0:
            return self.true_pos / self.n_pos
        return -1

    def get_tnr(self):
        """Computes true negative rate.
        """
        if self.n_neg > 0:
            return self.true_neg / self.n_neg
        return -1

    def get_acc(self):
        """Computes accuracy.
        """
        if self.n_pos + self.n_neg > 0:
            return (self.true_pos + self.true_neg) / (self.n_pos + self.n_neg)
        return -1

    def get_bacc(self):
        """Computes balanced accuracy.
        """
        recall = self.get_recall()
        tnr = self.get_tnr()
        if recall == -1 or tnr == -1:
            return -1
        return (recall + tnr) / 2.0

    def get_precision(self):
        """Computes precision.
        """
        n_pos_pred = self.true_pos + self.false_pos
        if n_pos_pred == 0:
            # Model makes 0 positive prediction.
            # This is a special case: we fall back to precision = 1 and recall = 0.
            return 1
        return self.true_pos / n_pos_pred

    def get_f1(self):
        """Computes F1.
        """
        recall = self.get_recall()
        precision = self.get_precision()
        
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        elif precision + recall == 0:
            return 0

        
    
    def to_dict(self):
        return {
            "class_name": self.class_name,
            "ap": self.get_ap(),
            "f1": self.get_f1(),
            "recall": self.get_recall(),
            "precision": self.get_precision(),
            "acc": self.get_acc()
        }


def build_language_model(model_dir):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_dir,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda"
    )
    return pipeline


def chat_tokenize(messages, pipeline):
    
    text_inputs = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    return text_inputs



def complete_parser(complete_result):
    
    try: 
        complete_result = complete_result.split("Augmented description: ")[1]
    except:
        pass
    
    return complete_result



def generation(messages, pipeline):
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    text_inputs = chat_tokenize(messages, pipeline)
    
    outputs = pipeline(
        text_inputs,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
    )

    return outputs[0]["generated_text"][len(text_inputs):]