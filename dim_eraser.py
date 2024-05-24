import fire
import random
import torch
import os.path as osp

from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from transformers import BlipProcessor, BlipForImageTextRetrieval
from fadata.utils import load_json, save_json, xywh_to_xyxy

from refine_prompts import *
from check_prompts import *
from utils import *


class BLIPScore:

    def __init__(
        self, 
        blip_model_path, 
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        torch_dtype=torch.float16
    ) -> None:

        self.device = device
        self.torch_dtype = torch_dtype
        self.processor = BlipProcessor.from_pretrained(blip_model_path)
        self.model = BlipForImageTextRetrieval.from_pretrained(blip_model_path, torch_dtype=torch_dtype).to(device)
        

    def __call__(self, images, texts, head='itm'):
        inputs = self.processor(images, texts, return_tensors="pt", truncation=True, padding=True).to(self.device, self.torch_dtype)
        with torch.no_grad():
            if head == 'itm':
                itm_scores = self.model(**inputs)[0].softmax(dim=-1)[:, 1]
                return itm_scores.cpu()
            else:
                cosine_scores = self.model(**inputs, use_itm_head=False)[0]
                return cosine_scores.cpu()



def main(
    llama_model_dir: str, 
    blip_model_dir: str,
    image_root: str, 
    f_check_result: str, 
    f_erase_result: str, 
    n_shot: int = 4,
    seed: int = 0,
):
    
    print(f"setting random seed to {seed}")
    setup_seed(seed)


    score_fn = BLIPScore(blip_model_dir)
        
    pipeline = build_language_model(llama_model_dir)

    data = load_json(f_check_result)['check_items']

    result_data = []

    for i, _item in enumerate(tqdm(data)):
        
        region_bbox = _item['region_anno']['bbox']
        xxyy_region_bbox = xywh_to_xyxy(deepcopy(region_bbox))
        
        image_path = osp.join(image_root, _item['image'])
        image = Image.open(image_path).convert("RGB")

        cropped_image = image.copy()
        cropped_image = cropped_image.crop(xxyy_region_bbox)

        check_dimensions, check_tuples = _item["check_dimensions"], _item["check_tuples"]
        
        erase_desps = []

        if len(check_dimensions) != 0:
        
            ori_desp = _item["answer"]
               
            global_score = score_fn([image], [ori_desp]).item()
            local_score = score_fn([cropped_image], [ori_desp]).item()

            if global_score > 0.9 and local_score > 0.9:
                continue

            erase_desps = [{
                "edit_item": "no_edit",
                "l_score": local_score,
                "g_score": global_score,
                "desp": ori_desp, 
            }]

            for (edim, etp) in zip(check_dimensions, check_tuples):
            
                erase_messages = [
                    {"role": "system", "content": erase_task_instruction}
                ]

                for example in random.sample(erase_few_shot_examples, n_shot):
                    
                    erase_messages.extend(
                        [
                            {"role": "user", "content": example['user']},
                            {"role": "assistant", "content": example['assistant']}
                        ]
                    )

                question = erase_question_template.format(dimension=f"{edim} {etp}", description=ori_desp)
                erase_messages.append({"role": "user", "content": question})

                edit_desp = generation(erase_messages, pipeline)

                e_global_score = score_fn([image], [edit_desp]).item()   
                e_local_score = score_fn([cropped_image], [edit_desp]).item()
                
                erase_desps.append({
                    "edit_item": "erase",
                    "l_score": e_local_score,
                    "g_score": e_global_score,
                    "desp": edit_desp, 
                    "edim": edim,
                    "etp": etp
                })
            
            _item["erase_desps"] = erase_desps
            result_data.append(_item)
        
        if i % 10 == 0 and i > 0:
            save_json(f_erase_result, result_data, indent=4)

    save_json(f_erase_result, result_data, indent=4)
    

if __name__ == "__main__":
    fire.Fire(main)
