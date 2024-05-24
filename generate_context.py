from collections import defaultdict
from fadata.utils import *



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


caption_annos = load_json("/content/annotations/captions_val2017.json")
attribute_annos = load_json("/content/ovad2000.json")


filtered_annos = []
atts = {}
for att in attribute_annos["attributes"]:
    atts[att["id"]] = att
    for k, value in atts.items():
        att_name = value['name']
        att_type, att_name = att_name.split(":")
        if att_type in TYPE2DIM:
            new_att_type = TYPE2DIM[att_type]
            atts[k]['name'] =  ":".join([new_att_type, att_name])


for anno in attribute_annos["annotations"]:
    _, _, w, h = anno['bbox']

    bbox_area = w * h
    if bbox_area >= 200:
        att_vec = anno["att_vec"]

        for index in range(len(att_vec)):
            att_name = atts[index]['name']
            att_type = att_name.split(":")[0]
            if att_type not in TYPE2DIM.values():
                att_vec[index] = -1

        if len(np.where(np.array(att_vec) == 1)[0]) > 3:
            filtered_annos.append(anno)

attribute_annos["annotations"] = filtered_annos

img_ann_map = defaultdict(list) # {image_id: [annotation]}
cat_img_map = defaultdict(list) # {category_id: [image_id]}
cat_ann_map = defaultdict(list) # {category_id: [instance_id]}
img_cap_map = defaultdict(list) # {image_id: [cap_annotation]}

anns = {} # {anno_id: annotation}
cats = {} # {category_id: category_info}
imgs = {} # {image_id: image_info}
atts = {} # {attribute_id: attribute_info}

oa_cooccurrence = {}

for ann in attribute_annos["annotations"]:
    img_ann_map[ann["image_id"]].append(ann)
    anns[ann["id"]] = ann


for img in attribute_annos["images"]:
    imgs[img["id"]] = img

for cat in attribute_annos["categories"]:
    cats[cat["id"]] = cat

for att in attribute_annos["attributes"]:
    atts[att["id"]] = att

for cap in caption_annos['annotations']:
    img_cap_map[cap["image_id"]].append(cap)




def generate_context(image_id, target_bbox, ctrl_dims=None):

    image_info = imgs[image_id]
    height = image_info['height']
    width = image_info['width']

    image_anns = img_ann_map[image_id]

    image_captions = ["[Context]"] + [item['caption'] for item in img_cap_map[image_id]]
    caption_context = " \n".join(image_captions)

    instances_context = []
    target_context = ""

    for instance_ann in image_anns:

        bbox = instance_ann['bbox']
        object_label = cats[instance_ann['category_id']]['name']
        attribute_labels = [atts[i]["name"] for i in range(len(instance_ann['att_vec'])) if instance_ann['att_vec'][i] == 1]

        if bbox == target_bbox:
            bbox = xywh_to_xyxy(bbox)

            bbox = convert_to_relative(bbox, (width, height))
            bbox = [round(n, 3) for n in bbox]
            target_context = f"[Target Region]\n{object_label} {bbox} {attribute_labels}"
        else:
            bbox = xywh_to_xyxy(bbox)
            bbox = convert_to_relative(bbox, (width, height))
            bbox = [round(n, 3) for n in bbox]
            # instances_context.append(f"{object_label} {bbox} {attribute_labels}")
            instances_context.append(f"{object_label} {bbox}")
    instances_context = "\n".join(instances_context)

    context = f"{caption_context}\n\n{instances_context}\n\n{target_context}"

    return context