DIMS = {
    'cleanliness': 0,
    'color': 1,
    'face expression': 2,
    'gender': 3,
    'hair type': 4,
    'length': 5,
    'material': 6,
    'maturity': 7,
    'pattern': 8,
    'pose': 9,
    'size': 10,
    'state': 11,
    'texture': 12,
    'transparency': 13
}

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

def check_parser(check_result):
    dims = []
    tps = []
    for line in check_result.splitlines():
        try:
            dim, tp = line.split(" (")
            if dim in DIMS:
                if tp.endswith(' '):
                    tp = tp[:-1]
                dims.append(dim)
                tps.append('(' + tp)
        except:
            continue
    return dims, tps



check_task_instruction = """Task: given input region description, decompose each description with dimension-specific tuples.
There are some predefined dimensions: [subject, object, cleanliness, color, face expression, gender, hair type, length, material, maturity, pattern, pose, size, state, texture, transparency]
Do not generate same tuples again. 
Do not generate tuples that are not explicitly described in the prompts.
Do not generate dimensions are not list in the predefined dimensions.
output format: predefined dimension tuple
"""

check_question_template = "input: {description} subject: {object_label}."

check_few_shot_examples = [
    {
        "user": "input: A plastic cup with a blue straw. subject: cup.",
        "assistant": "subject (cup) \nmaterial (cup, plastic) \nobject (straw) \ncolor (straw, blue)"
    },
    {
        "user": "input: A man is wearing a tie. subject: man." ,
        "assistant": "subject (man) \nobject (tie)"
    },
    {
        "user": "A red couch in the background. subject: couch.",
        "assistant": "subject (couch) \ncolor (couch, red) \nobject (background)"
    },

    {
        "user": "input: An iron bottle looks opaque. subject: bottle.",
        "assistant": "subject (bottle) \nmaterial (bottle, iron) \ntransparency (bottle, opaque)"
    },

    {
        "user": "input: This is a sink that is white in color. It is located in the bathroom and is quite visible. subject: sink.",
        "assistant": "subject (sink) \ncolor (sink, white) \nobject (bathroom) "
    },

    {
        "user": "input: There is a white couch in the living room. The couch is positioned behind a girl and a white chair. The girl is standing in front of the couch, and the white chair is located to her left. subject: couch.",
        "assistant": "subject (couch) \ncolor (couch, white) \nobject (living room) \nobject (girl) \npose (girl, stand) \nobject (chair) \ncolor (chair, white)"
    },

    {
        "user": "input: The man is wearing a red sweater and glasses. He is sitting and smiling at the camera. subject: man.",
        "assistant": "subject (man) \nface expression (man, smile) \npose (man, sit) \nobject (sweater) \ncolor (sweater, red) \nobject (glasses)"
    }
]