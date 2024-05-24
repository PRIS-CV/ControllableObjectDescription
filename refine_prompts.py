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


erase_task_instruction = "You are good at removing dimension relevant content from the given description. Each time I will give you a dimension and a tuple contain an object and its attributes of that dimension, you need to remove the relevant attribute content of the object in the description. Keep the sentence flowing without affecting the rest of the sentence"

erase_question_template = "Please remove {dimension} relevant content from this description: {description}"

erase_few_shot_examples = [
    {
        "user": "Please remove cleanliness (cabinet, clean) relevant content from this description: The cabinet is white in color, giving it a clean and classic appearance.",
        "assistant": "The cabinet is white in color, giving it a classic appearance."
    },

    {
        "user": "Please remove gender (women, female) relevant content from this description: The woman is comfortably seated on a couch. She has a yellow sweater on and her hand are clasped together, as if in prayer.",
        "assistant": "The person is comfortably seated on a couch. The person has a yellow sweater on and the person's hand are clasped together, as if in prayer."
    },

    {
        "user": "Please remove color (cabinet, white) relevant content from this description: The cabinet is white in color, giving it a clean and classic appearance.",
        "assistant": "The cabinet has a clean and classic appearance."
    },

    {
        "user": "Please remove face expression (woman, troubled) relevant content from this description: The woman is comfortably seated on a couch. She has a yellow sweater on and her hand are clasped together, as if in prayer. Her expression appears troubled, as if she is deeply engaged in thought or conversation.",
        "assistant": "The woman is comfortably seated on a couch. She has a yellow sweater on and her hand are clasped together, as if in prayer. She is deeply engaged in thought or conversation."
    },

    {
        "user": "Please remove state (kitchen scale, in use) relevant content from this description: The object is a black kitchen scale. It's used for weighing ingredients in the kitchen, which is essential for cooking and baking. The scale is in use for measuring ingredients.",
        "assistant": "The object is a black kitchen scale. It's used for weighing ingredients in the kitchen, which is essential for cooking and baking."
    },

    {
        "user": "Please remove pattern (shirt, checkered) relevant content from this description: The object is a shirt with a blue and white checkered pattern. This shirt is being worn by a man on the stage.",
        "assistant": "The object is a shirt with a blue and white color. This shirt is being worn by a man on the stage."
    },

    {
        "user": "Please remove pose (truck, facing left) relevant content from this description: The object is a truck that has a snow plow on its front. This suggests that the truck is used for snow removal or transportation during snowy conditions. The snow plow, located on the truck's left side, is a large, orange and black implement designed to clear snow from the road. The truck's front, which has the snow plow attached, is facing left, indicating it might be moving or clearing snow from a nearby area.",
        "assistant": "The object is a truck that has a snow plow on its front. This suggests that the truck is used for snow removal or transportation during snowy conditions. The snow plow, located on the truck's left side, is a large, orange and black implement designed to clear snow from the road. The truck might be moving or clearing snow from a nearby area."
    },

    {
        "user": "Please remove material (baseball bat, wooden) relevant content from this description: The object is a baseball bat. It's a wooden bat with a brown color. It's placed on a chair that is made of similar wood. The bat is positioned on the chair's seat, contributing to an aesthetically pleasing display.",
        "assistant": "The object is a baseball bat. It has a brown color. It's placed on a chair that is made of similar wood. The bat is positioned on the chair's seat, contributing to an aesthetically pleasing display."
    },

    {
        "user": "Please remove texture (table, rough) relevant content from this description: The table has a rough surface, and its color is red.",
        "assistant": "The table has a surface, and its color is red."
    },

    {
        "user": "Please remove hair type (woman, thick) relevant content from this description: Her hair is thick and curly, framing her face beautifully.",
        "assistant": "Her hair is curly, framing her face beautifully."
    },

    {
        "user": "Please remove maturity (couple, elderly) relevant content from this description: The elderly couple is walking hand in hand, displaying a mature love that has lasted for decades.",
        "assistant": "The couple is walking hand in hand, displaying a mature love that has lasted for decades."
    },

    {
        "user": "Please remove size (living room, spacious) relevant content from this description: The spacious living room is adorned with a large, plush sofa perfect for relaxing after a long day.",
        "assistant": "The living room is adorned with a large, plush sofa perfect for relaxing after a long day."
    },

    {
        "user": "Please remove transparency (windows, transparent) relevant content from this description: The glass windows offer a transparent view of the lush garden outside.",
        "assistant": "The glass windows offer a view of the lush garden outside."
    },

    {
        "user": "Please remove length (table, 6 feet in length) relevant content from this description: The table measures 6 feet in length and 3 feet in width, providing ample space for dining.",
        "assistant": "The table measures 3 feet in width, providing ample space for dining."
    }
]


complete_task_instruction = """You task is to augment object descriptions with additional dimension phrases. 
Each time the user will give you a original description and additional noisy dimension phrases (not all phrases are correct). You should try your best to choose the right additional phrases to enrich the original description.  
Keep the description flowing, don't make your own associations, just describe objective facts.
Output format: Augmented description: """


complete_few_shot_examples = [
    {
        "user": "Original description: The tv is quite large, taking up a significant part of the desk's space. It has a screen that is slightly lit up, indicating it's on. The tv is positioned near a keyboard, creating a typical desktop setup. Additional noisy dimension phrases: \nAdditional dimension phrases: \nmaterial: a tv is plastic\ntransparency: a tv is translucent\ncolor: a tv is white",
        "assistant": "Augmented description: The plastic tv is quite large, taking up a significant part of the desk's space. Its screen is slightly lit up, indicating it's on. The white tv is positioned near a keyboard, creating a typical desktop setup."
    }
]

complete_question_template = "Original description: {description} \nAdditional dimension phrases: \n{phrases}"