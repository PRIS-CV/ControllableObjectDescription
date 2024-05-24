import fire

from fadata.utils import *
from tqdm import tqdm

from check_prompts import *
from utils import *


def main(
    llama_model_dir: str, 
    eval_control: bool, 
    f_description: str, 
    f_check_result: str, 
    f_refined_description: str = None,
    only_check_erase: bool = False, 
    seed: int = 0
):
    
    setup_seed(seed)

    pipeline = build_language_model(llama_model_dir)

    data = load_json(f_description)
    
    if f_refined_description is not None:
        refined_data = load_json(f_refined_description)
        refine_qid_mapping = {_item["question_id"]: _item for _item in refined_data}
        
    
    result_summary = {
       'check_items': []
    }

    pred_labels = []
    gt_labels = []

    for i, _item in enumerate(tqdm(data)):

        question_id = _item["question_id"]
        
        if f_refined_description is not None and question_id in refine_qid_mapping:
            last_history = refine_qid_mapping[question_id]['edit_history'][-1]
            
            if only_check_erase and last_history['edit_item'] == 'complete':
                last_history = refine_qid_mapping[question_id]['edit_history'][-2]
                assert last_history['edit_item'] in ['erase', 'no_edit']

            _item['answer'] = last_history['desp']
            
        
            
        if eval_control:
            required_dims = _item['control_dims']
        else:
            required_dims = _item['dims']

        check_messages = [
            {"role": "system", "content": check_task_instruction}
        ]

        for example in check_few_shot_examples:
                    
            check_messages.extend(
                [
                    {"role": "user", "content": example['user']},
                    {"role": "assistant", "content": example['assistant']}
                ]
            )

        check_question = check_question_template.format(description = _item['answer'], object_label = _item['object_label'])

        check_messages.append({"role": "user", "content": check_question})

        check_result = generation(check_messages, pipeline)

        check_dimensions, check_tuples = check_parser(check_result)
        
        _item['check_dimensions'] = check_dimensions
        _item['check_tuples'] = check_tuples
        
        pred = [1 if _dim in check_dimensions else 0 for _dim in DIMS.keys()]
        gt = [1 if _dim in required_dims else 0 for _dim in DIMS.keys()]

        pred_labels.append(pred)
        gt_labels.append(gt)

        result_summary['check_items'].append(_item)

        if i % 10 == 0:
            save_json(f_check_result, result_summary, indent=4)


    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)

    class_metrics = {}
    for k, v in DIMS.items():
        class_metrics[k] = SingleClassMetric(k, pred_labels[:, v], gt_labels[:, v])
        result_summary[f"{k}_metric"] = class_metrics[k].to_dict()
    
    m_ap = round(sum([v.get_ap() for v in class_metrics.values()]) / len(class_metrics.values()), 4)
    m_recall = round(sum([v.get_recall() for v in class_metrics.values()]) / len(class_metrics.values()), 4)
    m_prec = round(sum([v.get_precision() for v in class_metrics.values()]) / len(class_metrics.values()), 4)
    m_f1 = round(sum([v.get_f1() for v in class_metrics.values()]) / len(class_metrics.values()), 4)
    
    result_summary["mAP"] = m_ap
    result_summary["mRecall"] = m_recall
    result_summary["mPrecision"] = m_prec
    result_summary["mF1"] = m_f1

    print(f"mAP: {m_ap}, mRecall: {m_recall}, mPrecision: {m_prec}, mF1: {m_f1}")
    save_json(f_check_result, result_summary, indent=4)
    print(f"Saving result to {f_check_result}")


if __name__ == "__main__":
    
    fire.Fire(main)
