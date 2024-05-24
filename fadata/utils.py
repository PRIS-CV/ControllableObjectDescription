from PIL import Image, ImageDraw
import numpy as np
import json
import typing as t
import torch
import torch.nn as nn

# https://en.wikipedia.org/wiki/YUV#SDTV_with_BT.601
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def bbox_expand(bbox:t.Sequence[int], image_size:t.Sequence[int], expand_ratio:float) -> t.Sequence[int]:
    r"""Expand the area of bounding box of object to gain more context.

        Args: 
            bbox (t.Sequence[int]): 
                The coordinates of bounding box [xmin, ymin, xmax, ymax].
            image_size (t.Sequence[int]):
                The width and height of image.
            expand_ratio (float):
                The expand ratio of bounding box.
        Return:
            t.Sequence[int, int, int, int]: The coordinates of expanded bounding box [xmin, ymin, xmax, ymax].
    """
    
    xmin, ymin, xmax, ymax = bbox
    wmax, hmax = image_size
    w, h = xmax - xmin, ymax - ymin
    margin = min(w, h) * expand_ratio * 0.5
    
    x1 = max(0, xmin - margin)
    y1 = max(0, ymin - margin)
    x2 = min(wmax, xmax + margin)
    y2 = min(hmax, ymax + margin)

    return x1, y1, x2, y2

def convert_to_relative(bbox:t.Sequence[int], image_size:t.Sequence[int]) -> t.Sequence[int]:
    """Convert a absolute bounding box (x, y, w, h) to bounding box
       
       bbox (Sequence): the sequence of bounding box coordinates, format: [<top-left-x>, <top-left-y>, <width>, <height>]
       image_size (Sequence): the sequence of image size, format: [width, height]
       
       return: a list of relative bounding box coordinates in [0, 1] in the following format: [<top-left-x>, <top-left-y>, <width>, <height>]
    """
    x, y, w, h = bbox
    w_img, h_img = image_size
    return [x / w_img, y / h_img, w / w_img, h / h_img]


def xyxy_to_xywh(bbox:t.Sequence[int]) -> t.Sequence[int]:
    x1, y1, x2, y2 = bbox
    return x1, y1, x2-x1, y2-y1


def xywh_to_xyxy(bbox:t.Sequence[int]) -> t.Sequence[int]:
    x, y, w, h = bbox
    return x, y, x + w, y + h

def relative_to_absolute(bbox: t.Sequence[float], w, h) -> t.Sequence[int]:
    x1, y1, x2, y2 = bbox
    x1 *= w
    x2 *= w
    y1 *= h
    y2 *= h
    return [x1, y1, x2, y2]


def get_bbox_type(bbox: t.Sequence):

    if any(isinstance(n, float) for n in bbox):
        return "relative"
    else:
        return "absolute"
    

def valid_bbox(bbox: t.Sequence):

    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return False

    if any(n < 0 for n in bbox):
        return False
    
    return True

def resize_bbox(bbox, ratios):
        r"""Resize the bounding box.
            Args:
                bbox (List[float, float, float, float]): 
                    The bounding box which format is [x1, y1, x2, y2].
                ratios (Tuple(float, float)):
                    (w_ratio, h_ratio) $w_ratio = origin_width / target_width$, $h_ratio = origin_height / target_height$
            Return:
                bbox (List[float, float, float, float]): 
                   The resized bounding box which format is [x1, y1, x2, y2].
        """

        xmin, ymin, xmax, ymax = bbox
        ratios_width, ratios_height = ratios
        
        xmin *= ratios_width
        xmax *= ratios_width
        ymin *= ratios_height
        ymax *= ratios_height

        return [xmin, ymin, xmax, ymax]

def polygon_to_mask(w, h, polygon):
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
    return mask

def load_json(file:str) -> t.Dict:
    r"""Load json data from file and convert it to dict.
        
        Args: 
            file (str): The path of json file.
        
        Return:
            t.Dict: The json data in a dict format.

    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_json(file:str, data:t.Dict, indent=None):
    r"""Load json data from file and convert it to dict.
        
        Args: 
            file (str): The path of json file.
        
        Return:
            t.Dict: The json data in a dict format.

    """
    
    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)

def get_stat(data:t.Dict):
    cnt_attr = {} # Number of attribute occurrences
    cnt_obj = {} # Number of object occurrences
    cnt_pair = {} # Number of pair occurrences
    cooc = {} # co-occurrences of attributes (cooc['red']['blue'] is the number of times red and blue appear together)

    obj_afford = {}
    obj_afford_cooc = {}
    
    n_images = 0

    for ins in data:
        o = ins['object_name']
        box = ins['instance_bbox']
            
        n_images += 1

        if o not in cnt_obj:
            cnt_obj[o] = 0
            obj_afford[o] = {}
            obj_afford_cooc[o] = {}

        cnt_obj[o] += 1

        for a in set(ins['positive_attributes']): # possible duplicates so we use set
            if a not in cnt_attr:
                cnt_attr[a] = 0
                cooc[a] = {}
            cnt_attr[a] += 1

            pair = (a, o)
            if pair not in cnt_pair:
                cnt_pair[pair] = 0
            cnt_pair[pair] += 1

            if a not in obj_afford[o]:
                obj_afford[o][a] = 0
                obj_afford_cooc[o][a] = {}
            obj_afford[o][a] += 1

            for other_a in set(ins['positive_attributes']):
                if a != other_a:
                    if other_a not in cooc[a]:
                        cooc[a][other_a] = 0
                    cooc[a][other_a] += 1

                    if other_a not in obj_afford_cooc[o][a]:
                        obj_afford_cooc[o][a][other_a] = 0
                    obj_afford_cooc[o][a][other_a] += 1
                    
    return cnt_attr, cnt_obj, cnt_pair, cooc, obj_afford, obj_afford_cooc, n_images



class UnNormalize(nn.Module):
    
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        
        self.mean = torch.tensor(mean).view((-1, 1, 1))
        self.std = torch.tensor(std).view((-1, 1, 1))

    def __call__(self,x):
        
        x = (x * self.std) + self.mean
        return torch.clip(x, 0, None)


def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def convert_image_to_rgb(image, format):
    """
    Convert an image from given format to RGB.

    Args:
        image (np.ndarray or Tensor): an HWC image
        format (str): the format of input image, also see `read_image`

    Returns:
        (np.ndarray): (H,W,3) RGB image in 0-255 range, can be either float or uint8
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if format == "BGR":
        image = image[:, :, [2, 1, 0]]
    elif format == "YUV-BT.601":
        image = np.dot(image, np.array(_M_YUV2RGB).T)
        image = image * 255.0
    else:
        if format == "L":
            image = image[:, :, 0]
        image = image.astype(np.uint8)
        image = np.asarray(Image.fromarray(image, mode=format).convert("RGB"))
    return image

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR" or "YUV-BT.601".

    Returns:
        image (np.ndarray):
            an HWC image in the given format, which is 0-255, uint8 for
            supported image modes in PIL or "BGR"; float (0-1 for Y) for YUV-BT.601.
    """
    with open(file_name, "rb") as f:
        image = Image.open(f)

        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        image = _apply_exif_orientation(image)
        return convert_PIL_to_numpy(image, format)


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color
