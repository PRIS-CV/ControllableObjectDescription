import fire
import random
import torch
import transformers
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


class AttributeClassifier:
    
    templates_dict = {
        "has": {
            "none": ["{attr} {dobj} {noun}"],
            "a": ["a {attr} {dobj} {noun}", "a {noun} has {attr} {dobj}"],
            "the": ["the {attr} {dobj} {noun}", "the {noun} has {attr} {dobj}"],
            "photo": [
                "a photo of a {attr} {dobj} {noun}",
                "a photo of an {noun} which has {attr} {dobj}",
                "a photo of the {attr} {dobj} {noun}",
                "a photo of the {noun} which has {attr} {dobj}",
            ],
        },
        "is": {
            "none": ["{attr} {noun}"],
            "a": ["a {attr} {noun}", "a {noun} is {attr}"],
            "the": ["the {attr} {noun}", "the {noun} is {attr}"],
            "photo": [
                "a photo of a {attr} {noun}",
                "a photo of a {noun} which is {attr}",
                "a photo of the {attr} {noun}",
                "a photo of the {noun} which is {attr}",
            ],
        }
    }


    def __init__(self, score_fn, attributes_data, batch_size, obj_dim_attr_comb=None) -> None:
        self.score_fn = score_fn
        self.attributes_data = attributes_data
        self.obj_dim_attr_comb = obj_dim_attr_comb
        self.batch_size = batch_size

    def build_template(self, object_word: str, dim: str, prompt_att: str = 'a'):

        if prompt_att in self.templates_dict["is"].keys():
            use_prompts = [prompt_att]
        else:  # use all prompts
            use_prompts = ["a", "the", "none"]
        
        all_att_templates = []
        for att_dict in self.attributes_data:
            att_w_type = att_dict["name"]
            att_type, att_list = att_w_type.split(":")
            
            if self.obj_dim_attr_comb is not None:
                att_list = att_list.split("/")
                tp = TYPE2DIM[dim]
                comb = self.obj_dim_attr_comb[object_word][tp]
                att_list = "/".join([att for att in att_list if att in comb])
                
                if att_list == "":
                    continue

            if att_type != dim:
                continue
            assert att_type == att_dict["type"]
            is_has = att_dict["is_has_att"]

            dobj_name = (
                att_type.replace(" tone", "")
            )

            # extend the maturity to include other words
            if att_list == "young/baby":
                att_list += "/kid/kids/child/toddler/boy/girl"
            elif att_list == "adult/old/aged":
                att_list += "/teen/elder"

            att_templates = []
            for syn in att_list.split("/"):
                for prompt in use_prompts:
                    for template in self.templates_dict[is_has][prompt]:
                        if is_has == "has":
                            att_templates.append(
                                template.format(
                                    attr=syn, dobj=dobj_name, noun=object_word
                                ).strip()
                            )
                        elif is_has == "is":
                            att_templates.append(
                                template.format(attr=syn, noun=object_word).strip()
                            )
            all_att_templates.append(att_templates)
        
        return all_att_templates
    
    def __call__(self, image: Image.Image, object_label: str, dims: list):
        
        complete_phrases = []
        complete_confidences = []

        dim2type = {v:k for k, v in TYPE2DIM.items()}

        for dim in dims:
            dim_templates = self.build_template(object_word=object_label, dim=dim2type[dim])
            templates_texts = []
            for temps in dim_templates:
                templates_texts.extend(temps)
    
            dim_scores = []
            
            for batch_index in range(0, len(templates_texts), self.batch_size):
                batch_texts = templates_texts[batch_index: batch_index + self.batch_size]
                scores = self.score_fn([image] * len(batch_texts), batch_texts)
                dim_scores.append(scores)

            if len(dim_scores) != 0:

                dim_scores = torch.cat(dim_scores)
                max_value = torch.max(dim_scores).item()
                max_index = torch.argmax(dim_scores).item()
                
                max_score_words = templates_texts[max_index]
                complete_confidences.append(max_value)
                complete_phrases.append(max_score_words)

        return complete_phrases, complete_confidences

        




def main(
    llama_model_dir: str, 
    blip_model_dir: str,
    image_root: str, 
    f_description: str,
    f_result: str, 
    f_check_result: str = None,
    f_ovad_anno: str = None, 
    complete: bool = False,
    refine_control: bool = False,
    complete_threshold: float = 0.5,
    batch_size: int = 4, 
    n_shot: int = 4,
    g_threshold: float = 0,
    l_threshold: float = 0.1,
    save_steps: int = 1, 
    f_obj_dim_attr_comb: str = None, 
    seed: int = 0, 
    f_update_refined_description: str = None
):
    print(f"setting random seed to {seed}")
    setup_seed(seed)

    print(f"refine_control: {refine_control}")
    print(f"complete: {complete}")
    
    score_fn = BLIPScore(blip_model_dir)
    
    if f_obj_dim_attr_comb is not None:
        obj_dim_attr_comb = load_json(f_obj_dim_attr_comb)
    else:
        obj_dim_attr_comb = None

    if complete: 
        completer = AttributeClassifier(score_fn, load_json(f_ovad_anno)["attributes"], batch_size, obj_dim_attr_comb=obj_dim_attr_comb)
        

    pipeline = build_language_model(llama_model_dir)

    if f_check_result is not None:
        check_data = load_json(f_check_result)['check_items']
    else:
        check_data = None

    if f_update_refined_description is not None:
        data = load_json(f_update_refined_description)
    else:
        data = load_json(f_description)

    result_data = []

    for i, _item in enumerate(tqdm(data)):
        
        region_bbox = _item['region_anno']['bbox']
        xxyy_region_bbox = xywh_to_xyxy(deepcopy(region_bbox))
        
        image_path = osp.join(image_root, _item['image'])
        image = Image.open(image_path).convert("RGB")

        cropped_image = image.copy()
        cropped_image = cropped_image.crop(xxyy_region_bbox)


        object_label = _item["object_label"]
        all_dims = _item["dims"].split(', ')

        if refine_control:
            control_dims = _item["control_dims"].split(', ')
        else:
            control_dims = []
        
        
        """ Check_dimension
        """

        if check_data is not None:
            check_dimensions = check_data[i]['check_dimensions']
            check_tuples = check_data[i]['check_tuples']
        
        else:
        
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

        if f_update_refined_description is not None:
            
            if _item['edit_history'][-1]['edit_item'] == 'complete':
                edit_history = _item['edit_history'][:-1]
            else:
                edit_history = _item['edit_history']
            
            erased_dims = [eh['tuple'].split(' (')[0] for eh in edit_history if eh['edit_item'] == 'erase']
            cur_desp = edit_history[-1]['desp']
        
        else:
            erased_dims = []
            edit_history = []

            cur_desp = _item["answer"]
            

            if len(check_dimensions) != 0:
            

                global_score = score_fn([image], [cur_desp]).item()
                local_score = score_fn([cropped_image], [cur_desp]).item()

                if global_score > 0.9 and local_score > 0.9:
                    continue

                edit_history = [{
                    "edit_item": "no_edit",
                    "l_score": local_score,
                    "g_score": global_score,
                    "desp": cur_desp, 
                    "tuple": ""
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

                    question = erase_question_template.format(dimension=f"{edim} {etp}", description=cur_desp)
                    erase_messages.append({"role": "user", "content": question})

                    edit_desp = generation(erase_messages, pipeline)

                    e_global_score = score_fn([image], [edit_desp]).item()   
                    e_local_score = score_fn([cropped_image], [edit_desp]).item()

                    if refine_control and edim not in control_dims:

                        cur_desp = edit_desp
                        global_score = e_global_score
                        local_score = e_local_score
                        edit_history.append({
                            "edit_item": "erase",
                            "l_score": local_score,
                            "g_score": global_score,
                            "desp": cur_desp, 
                            "tuple": f"{edim} {etp}"
                        })

                        erased_dims.append(edim)
                        continue
                        
                    if ((e_global_score - global_score) > g_threshold) and ((e_local_score - local_score) > l_threshold):
                        
                        cur_desp = edit_desp
                        global_score = e_global_score
                        local_score = e_local_score
                        edit_history.append({
                            "edit_item": "erase",
                            "l_score": local_score,
                            "g_score": global_score,
                            "desp": cur_desp, 
                            "tuple": f"{edim} {etp}"
                        })
                        erased_dims.append(edim)
                    else:
                        continue

        """ Complete
        """

        if complete:
            if refine_control:
                erased_control_dims = set(control_dims).difference(set(erased_dims))
                complete_dims = list(set(control_dims).difference(check_dimensions).union(erased_control_dims))
            else:
                if f_obj_dim_attr_comb is None:
                    complete_dims = list(set(all_dims).difference(check_dimensions).union(set(erased_dims)))
                else:
                    complete_dims = list(set(all_dims).difference(check_dimensions).union(set(erased_dims).intersection(all_dims)))
            
            complete_phrases, confidences = completer(cropped_image, object_label=object_label, dims=complete_dims)
            
            filtered_complete_phrases = []
            filtered_confidences = []

            for cdim, phrase, conf in zip(complete_dims, complete_phrases, confidences):
                if conf > complete_threshold:
                    filtered_complete_phrases.append(f"{cdim}: {phrase}")
                    filtered_confidences.append(conf)
            
            if len(filtered_complete_phrases) > 0:

                complete_messages = [
                    {"role": "system", "content": complete_task_instruction}
                ]

                question = complete_question_template.format(description=cur_desp, phrases=" \n".join(filtered_complete_phrases))
                complete_messages.append({"role": "user", "content": question})
                
                complete_desp = complete_parser(generation(complete_messages, pipeline))


                global_score = score_fn([image], [complete_desp]).item()
                local_score = score_fn([cropped_image], [complete_desp]).item()

                edit_history.append({
                    "edit_item": "complete",
                    "l_score": local_score,
                    "g_score": global_score,
                    "desp": complete_desp, 
                    "complete_phrases": filtered_complete_phrases, 
                    "confidences": filtered_confidences, 
                })
            
        if len(edit_history) > 1:
            copy_item = deepcopy(_item)
            copy_item["edit_history"] = edit_history
            result_data.append(copy_item)
            
        if len(result_data) % save_steps == 0 and len(result_data) > 0:
            print(f"Saving intermediate result to {f_result}")
            save_json(f_result, result_data, indent=4)

    save_json(f_result, result_data, indent=4)
                

if __name__ == "__main__":
    fire.Fire(main)
