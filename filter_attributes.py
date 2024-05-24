from collections import defaultdict
import fire
import os.path as osp

from pprint import pprint
from PIL import Image
from tqdm import tqdm
from copy import deepcopy
from fadata.utils import load_json, save_json

from refine_prompts import *
from check_prompts import *
from utils import *


filter_task_instruction = """
You task is to select the attribute tags can be used to describe the object in the real world. 
Each time the user will give an object name and a attribute dimenion and all attributes tags of the dimension.
Please choose from the labels provided and do not make up your own tag. 
"""

filter_question_template = "object name: {object_label}, dimension: {dim}, tags: {tags}"

filter_few_shot_examples = [
    {
        "user": "object name: tv, dimension: cleanliness, tags: clean, neat, unclean, dirt, dirty, muddy",
        "assistant": "clean, neat, unclean, dirt, dirty, muddy"
    },
    {
        "user": "object name: bus, dimension: material, tags: asphalt, cement, clay, concrete, stucco, ceramic, brick, porcelain, glass, leather, metal, metallic, aluminum, brass, copper-zinc, iron, stainless steel, steel, silver, paper, cardboard, polymers, plastic, rubber, styrofoam, polymer, stone, granite, cobblestone, gravel, marble, pebbled, rocky, sandy, textile, cloth, fabric, denim, cotton, jean, silk, plush, wood, wooden, bamboo, hardwood",
        "assistant": "iron, copper-zinc, steel, silver, aluminum, stainless steel, metallic, metal, brass"
    }
]


def filter_parser(result):
    if "null" not in result:
        return result.split(', ')
    else:
        return "null"


TYPE2DIM = {
    'cleanliness': 'cleanliness', 
    'color': 'color', 
    'face expression': 'face expression', 
    'gender': 'gender', 
    'hair type': 'hair type', 
    'length': 'length', 
    'material': 'material', 
    'maturity': 'maturity', 
    'pattern': 'pattern', 
    'position': 'pose', 
    'size': 'size', 
    'state': 'state', 
    'texture': 'texture', 
    'optical property': 'transparency', 
}

class AttributeTagFilter:
    
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
            "tag": ["{attr}"]
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
            "tag": ["{attr}"]
        }
    }


    def __init__(self, attributes_data) -> None:
        self.attributes_data = attributes_data

    def build_template(self, object_word: str, dim: str, prompt_att: str = 'tag'):

        if prompt_att in self.templates_dict["is"].keys():
            use_prompts = [prompt_att]
        else:  # use all prompts
            use_prompts = ["a", "the", "none"]
        
        all_att_templates = []
        for att_dict in self.attributes_data:
            att_w_type = att_dict["name"]
            att_type, att_list = att_w_type.split(":")
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

        




def main(
    llama_model_dir: str, 
    f_question: str, 
    f_result: str, 
    f_ovad_anno: str = None, 
    
):
    
    
    question_items = load_json(f_question)

    obj_dim_attr_comb = defaultdict(dict)
    for _item in question_items:
        for attribute_label in _item['attribute_label']:
            attribute_dim, labels = attribute_label.split(':')
            labels = labels.split('/')
            if attribute_dim in obj_dim_attr_comb[_item['object_label']]:
                obj_dim_attr_comb[_item['object_label']][attribute_dim].extend(labels)
            else:
                obj_dim_attr_comb[_item['object_label']][attribute_dim] = labels
            obj_dim_attr_comb[_item['object_label']][attribute_dim] = list(set(obj_dim_attr_comb[_item['object_label']][attribute_dim]))
    obj_dim_attr_comb = dict(obj_dim_attr_comb)

    save_json("gt_obj_dim_attr_comb.json", obj_dim_attr_comb, indent=4)

    exit()

    object_labels = set([_item['object_label'] for _item in question_items])
    dims = list(DIMS.keys())

    dim2type = {v:k for k, v in TYPE2DIM.items()}
    
    tag_filter = AttributeTagFilter(load_json(f_ovad_anno)["attributes"])

    pipeline = build_language_model(llama_model_dir)

    result_data = defaultdict(dict)

    # from IPython import embed; embed()

    for object_label in tqdm(object_labels):

        for dim in dims:

            if dim not in obj_dim_attr_comb[object_label].keys():
                continue
            
            filter_messages = [
                {"role": "system", "content": filter_task_instruction}
            ]

            for example in filter_few_shot_examples:
                        
                filter_messages.extend(
                    [
                        {"role": "user", "content": example['user']},
                        {"role": "assistant", "content": example['assistant']}
                    ]
                )
            
            dim_tags = tag_filter.build_template(object_label, dim=dim2type[dim])

            flatten_dim_tags = []
            for tags in dim_tags:
                flatten_dim_tags.extend(tags)
            dim_tags = flatten_dim_tags
            
            question = filter_question_template.format(object_label=object_label, dim=dim, tags=", ".join(dim_tags))

            filter_messages.append({"role": "user", "content": question})

            filter_result = generation(filter_messages, pipeline)

            valid_tags = filter_parser(filter_result)

            result_data[object_label][dim] = valid_tags
            print(question)
            print(valid_tags)
            # from IPython import embed; embed()

    save_json("obj_attr_combinations.json", dict(result_data), indent=4)

    

if __name__ == "__main__":
    fire.Fire(main)
