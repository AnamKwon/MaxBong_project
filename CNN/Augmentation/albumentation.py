from __future__ import absolute_import
from __future__ import division
import functions as F
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from abc import ABCMeta, abstractmethod
import sys

def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""

    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    return wrapper
def normalize_bbox(bbox, rows, cols):
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.
    Args:
        bbox (tuple): Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: Normalized bounding box `(x_min, y_min, x_max, y_max)`.
    Raises:
        ValueError: If rows or cols is less or equal zero
    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows

    return (x_min, y_min, x_max, y_max) + tail


def denormalize_bbox(bbox, rows, cols):
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.
    Args:
        bbox (tuple): Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
    Raises:
        ValueError: If rows or cols is less or equal zero
    """
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return (x_min, y_min, x_max, y_max) + tail


def normalize_bboxes(bboxes, rows, cols):
    """Normalize a list of bounding boxes.
    Args:
        bboxes (List[tuple]): Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        List[tuple]: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
    """
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]
def calculate_bbox_area(bbox, rows, cols):
    """Calculate the area of a bounding box in pixels.
    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
    Return:
        int: Area of a bounding box in pixels.
    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    area = (x_max - x_min) * (y_max - y_min)
    return area

def normalize_bboxes(bboxes, rows, cols):
    """Normalize a list of bounding boxes.
    Args:
        bboxes (List[tuple]): Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        List[tuple]: Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
    """
    return [normalize_bbox(bbox, rows, cols) for bbox in bboxes]


def denormalize_bboxes(bboxes, rows, cols):
    """Denormalize a list of bounding boxes.
    Args:
        bboxes (List[tuple]): Normalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        List[tuple]: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max)]`.
    """
    return [denormalize_bbox(bbox, rows, cols) for bbox in bboxes]
def convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity=False):
    """Convert a bounding box from a format specified in `source_format` to the format used by albumentations:
    normalized coordinates of bottom-left and top-right corners of the bounding box in a form of
    `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.
    Args:
        bbox (tuple): A bounding box tuple.
        source_format (str): format of the bounding box. Should be 'coco', 'pascal_voc', or 'yolo'.
        check_validity (bool): Check if all boxes are valid boxes.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.
    Note:
        The `coco` format of a bounding box looks like `(x_min, y_min, width, height)`, e.g. (97, 12, 150, 200).
        The `pascal_voc` format of a bounding box looks like `(x_min, y_min, x_max, y_max)`, e.g. (97, 12, 247, 212).
        The `yolo` format of a bounding box looks like `(x, y, width, height)`, e.g. (0.3, 0.1, 0.05, 0.07);
        where `x`, `y` coordinates of the center of the box, all values normalized to 1 by image height and width.
    Raises:
        ValueError: if `target_format` is not equal to `coco` or `pascal_voc`, ot `yolo`.
        ValueError: If in YOLO format all labels not in range (0, 1).
    """
    if source_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            "Unknown source_format {}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'".format(source_format)
        )
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()

    if source_format == "coco":
        (x_min, y_min, width, height), tail = bbox[:4], tuple(bbox[4:])
        x_max = x_min + width
        y_max = y_min + height
    elif source_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        bbox, tail = bbox[:4], tuple(bbox[4:])
        _bbox = np.array(bbox[:4])
        if np.any((_bbox <= 0) | (_bbox > 1)):
            raise ValueError("In YOLO format all labels must be float and in range (0, 1]")

        x, y, width, height = denormalize_bbox(bbox, rows, cols)

        x_min = int(x - width / 2 + 1)
        x_max = int(x_min + width)
        y_min = int(y - height / 2 + 1)
        y_max = int(y_min + height)
    else:
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    bbox = (x_min, y_min, x_max, y_max) + tail
    bbox = normalize_bbox(bbox, rows, cols)
    if check_validity:
        check_bbox(bbox)
    return bbox

def convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity=False):
    """Convert a bounding box from the format used by albumentations to a format, specified in `target_format`.
    Args:
        bbox (tuple): An albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        target_format (str): required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows (int): Image height.
        cols (int): Image width.
        check_validity (bool): Check if all boxes are valid boxes.
    Returns:
        tuple: A bounding box.
    Note:
        The `coco` format of a bounding box looks like `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
        The `pascal_voc` format of a bounding box looks like `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
        The `yolo` format of a bounding box looks like `[x, y, width, height]`, e.g. [0.3, 0.1, 0.05, 0.07].
    Raises:
        ValueError: if `target_format` is not equal to `coco`, `pascal_voc` or `yolo`.
    """
    if target_format not in {"coco", "pascal_voc", "yolo"}:
        raise ValueError(
            "Unknown target_format {}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'".format(target_format)
        )
    if check_validity:
        check_bbox(bbox)
    bbox = denormalize_bbox(bbox, rows, cols)
    if target_format == "coco":
        (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])
        width = x_max - x_min
        height = y_max - y_min
        bbox = (x_min, y_min, width, height) + tail
    elif target_format == "yolo":
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/scripts/voc_label.py#L12
        (x_min, y_min, x_max, y_max), tail = bbox[:4], bbox[4:]
        x = int((x_min + x_max) / 2 - 1)
        y = int((y_min + y_max) / 2 - 1)
        width = x_max - x_min
        height = y_max - y_min
        bbox = normalize_bbox((x, y, width, height) + tail, rows, cols)
    return bbox


def convert_bboxes_to_albumentations(bboxes, source_format, rows, cols, check_validity=False):
    """Convert a list bounding boxes from a format specified in `source_format` to the format used by albumentations
    """
    return [convert_bbox_to_albumentations(bbox, source_format, rows, cols, check_validity) for bbox in bboxes]


def convert_bboxes_from_albumentations(bboxes, target_format, rows, cols, check_validity=False):
    """Convert a list of bounding boxes from the format used by albumentations to a format, specified
    in `target_format`.
    Args:
        bboxes (List[tuple]): List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        target_format (str): required format of the output bounding box. Should be 'coco', 'pascal_voc' or 'yolo'.
        rows (int): Image height.
        cols (int): Image width.
        check_validity (bool): Check if all boxes are valid boxes.
    Returns:
        list[tuple]: List of bounding box.
    """
    return [convert_bbox_from_albumentations(bbox, target_format, rows, cols, check_validity) for bbox in bboxes]


def check_bbox(bbox):
    """Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for name, value in zip(["x_min", "y_min", "x_max", "y_max"], bbox[:4]):
        if not 0 <= value <= 1:
            raise ValueError(
                "Expected {name} for bbox {bbox} "
                "to be in the range [0.0, 1.0], got {value}.".format(bbox=bbox, name=name, value=value)
            )
    x_min, y_min, x_max, y_max = bbox[:4]
    if x_max <= x_min:
        raise ValueError("x_max is less than or equal to x_min for bbox {bbox}.".format(bbox=bbox))
    if y_max <= y_min:
        raise ValueError("y_max is less than or equal to y_min for bbox {bbox}.".format(bbox=bbox))


def check_bboxes(bboxes):
    """Check if bboxes boundaries are in range 0, 1 and minimums are lesser then maximums"""
    for bbox in bboxes:
        check_bbox(bbox)


def filter_bboxes(bboxes, rows, cols, min_area=0.0, min_visibility=0.0):
    """Remove bounding boxes that either lie outside of the visible area by more then min_visibility
    or whose area in pixels is under the threshold set by `min_area`. Also it crops boxes to final image size.
    Args:
        bboxes (List[tuple]): List of albumentation bounding box `(x_min, y_min, x_max, y_max)`.
        rows (int): Image height.
        cols (int): Image width.
        min_area (float): Minimum area of a bounding box. All bounding boxes whose visible area in pixels.
            is less than this value will be removed. Default: 0.0.
        min_visibility (float): Minimum fraction of area for a bounding box to remain this box in list. Default: 0.0.
    Returns:
        List[tuple]: List of bounding box.
    """
    resulting_boxes = []
    for bbox in bboxes:
        transformed_box_area = calculate_bbox_area(bbox, rows, cols)
        bbox, tail = tuple(np.clip(bbox[:4], 0, 1.0)), tuple(bbox[4:])
        clipped_box_area = calculate_bbox_area(bbox, rows, cols)
        if not transformed_box_area or clipped_box_area / transformed_box_area <= min_visibility:
            continue
        else:
            bbox = tuple(np.clip(bbox[:4], 0, 1.0))
        if calculate_bbox_area(bbox, rows, cols) <= min_area:
            continue
        resulting_boxes.append(bbox + tail)
    return resulting_boxes



class DataProcessor:
    def __init__(self, params, additional_targets=None):
        self.params = params
        self.data_fields = [self.default_data_name]
        if additional_targets is not None:
            for k, v in additional_targets.items():
                if v == self.default_data_name:
                    self.data_fields.append(k)

    @property
    @abstractmethod
    def default_data_name(self):
        raise NotImplementedError

    def ensure_data_valid(self, data):
        pass

    def ensure_transforms_valid(self, transforms):
        pass

    def postprocess(self, data):
        rows, cols = data["image"].shape[:2]

        for data_name in self.data_fields:
            data[data_name] = self.filter(data[data_name], rows, cols)
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="from")

        data = self.remove_label_fields_from_data(data)
        return data

    def preprocess(self, data):
        data = self.add_label_fields_to_data(data)

        rows, cols = data["image"].shape[:2]
        for data_name in self.data_fields:
            data[data_name] = self.check_and_convert(data[data_name], rows, cols, direction="to")

    def check_and_convert(self, data, rows, cols, direction="to"):
        if self.params.format == "albumentations":
            self.check(data, rows, cols)
            return data

        if direction == "to":
            return self.convert_to_albumentations(data, rows, cols)

        return self.convert_from_albumentations(data, rows, cols)

    @abstractmethod
    def filter(self, data, rows, cols):
        pass

    @abstractmethod
    def check(self, data, rows, cols):
        pass

    @abstractmethod
    def convert_to_albumentations(self, data, rows, cols):
        pass

    @abstractmethod
    def convert_from_albumentations(self, data, rows, cols):
        pass

    def add_label_fields_to_data(self, data):
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            for field in self.params.label_fields:
                assert len(data[data_name]) == len(data[field])
                data_with_added_field = []
                for d, field_value in zip(data[data_name], data[field]):
                    data_with_added_field.append(list(d) + [field_value])
                data[data_name] = data_with_added_field
        return data

    def remove_label_fields_from_data(self, data):
        if self.params.label_fields is None:
            return data
        for data_name in self.data_fields:
            label_fields_len = len(self.params.label_fields)
            for idx, field in enumerate(self.params.label_fields):
                field_values = []
                for bbox in data[data_name]:
                    field_values.append(bbox[-label_fields_len + idx])
                data[field] = field_values
            if label_fields_len:
                data[data_name] = [d[:-label_fields_len] for d in data[data_name]]
        return data

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY3:
    string_types = (str,)
else:
    string_types = (basestring,)  # noqa: F821


def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""

    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get("__slots__")
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop("__dict__", None)
        orig_vars.pop("__weakref__", None)
        return metaclass(cls.__name__, cls.__bases__, orig_vars)

    return wrapper

REPR_INDENT_STEP = 2


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms
        self.start_end = self._find_dual_start_end(transforms)

    def _find_dual_start_end(self, transforms):
        dual_start_end = None
        last_dual = None
        for idx, transform in enumerate(transforms):
            if isinstance(transform, DualTransform):
                last_dual = idx
                if dual_start_end is None:
                    dual_start_end = [idx]
            if isinstance(transform, BaseCompose):
                inside = self._find_dual_start_end(transform)
                if inside is not None:
                    last_dual = idx
                    if dual_start_end is None:
                        dual_start_end = [idx]
        if dual_start_end is not None:
            dual_start_end.append(last_dual)
        return dual_start_end

    def get_always_apply(self, transforms):
        new_transforms = []
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                new_transforms.extend(self.get_always_apply(transform))
            elif transform.always_apply:
                new_transforms.append(transform)
        return Transforms(new_transforms)

    def __getitem__(self, item):
        return self.transforms[item]


def set_always_apply(transforms):
    for t in transforms:
        t.always_apply = True
SERIALIZABLE_REGISTRY = {}
class SerializableMeta(type):
    """
    A metaclass that is used to register classes in `SERIALIZABLE_REGISTRY` so they can be found later
    while deserializing transformation pipeline using classes full names.
    """

    def __new__(cls, name, bases, class_dict):
        cls_obj = type.__new__(cls, name, bases, class_dict)
        SERIALIZABLE_REGISTRY[cls_obj.get_class_fullname()] = cls_obj
        return cls_obj


def to_dict(transform, on_not_implemented_error="raise"):
    """
    Take a transform pipeline and convert it to a serializable representation that uses only standard
    python data types: dictionaries, lists, strings, integers, and floats.
    Args:
        transform (object): A transform that should be serialized. If the transform doesn't implement the `to_dict`
            method and `on_not_implemented_error` equals to 'raise' then `NotImplementedError` is raised.
            If `on_not_implemented_error` equals to 'warn' then `NotImplementedError` will be ignored
            but no transform parameters will be serialized.
    """
    if on_not_implemented_error not in {"raise", "warn"}:
        raise ValueError(
            "Unknown on_not_implemented_error value: {}. Supported values are: 'raise' and 'warn'".format(
                on_not_implemented_error
            )
        )
    try:
        transform_dict = transform._to_dict()  # skipcq: PYL-W0212
    except NotImplementedError as e:
        if on_not_implemented_error == "raise":
            raise e

        transform_dict = {}
        warnings.warn(
            "Got NotImplementedError while trying to serialize {obj}. Object arguments are not preserved. "
            "Implement either '{cls_name}.get_transform_init_args_names' or '{cls_name}.get_transform_init_args' "
            "method to make the transform serializable".format(obj=transform, cls_name=transform.__class__.__name__)
        )
    return {"__version__": __version__, "transform": transform_dict}


def instantiate_lambda(transform, lambda_transforms=None):
    if transform.get("__type__") == "Lambda":
        name = transform["__name__"]
        if lambda_transforms is None:
            raise ValueError(
                "To deserialize a Lambda transform with name {name} you need to pass a dict with this transform "
                "as the `lambda_transforms` argument".format(name=name)
            )
        transform = lambda_transforms.get(name)
        if transform is None:
            raise ValueError("Lambda transform with {name} was not found in `lambda_transforms`".format(name=name))
        return transform
    return None


def from_dict(transform_dict, lambda_transforms=None):
    """
    Args:
        transform (dict): A dictionary with serialized transform pipeline.
        lambda_transforms (dict): A dictionary that contains lambda transforms, that is instances of the Lambda class.
            This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
            in that dictionary should be named same as `name` arguments in respective lambda transforms from
            a serialized pipeline.
    """
    register_additional_transforms()
    transform = transform_dict["transform"]
    lmbd = instantiate_lambda(transform, lambda_transforms)
    if lmbd:
        return lmbd
    name = transform["__class_fullname__"]
    args = {k: v for k, v in transform.items() if k != "__class_fullname__"}
    cls = SERIALIZABLE_REGISTRY[name]
    if "transforms" in args:
        args["transforms"] = [
            from_dict({"transform": t}, lambda_transforms=lambda_transforms) for t in args["transforms"]
        ]
    return cls(**args)


def check_data_format(data_format):
    if data_format not in {"json", "yaml"}:
        raise ValueError("Unknown data_format {}. Supported formats are: 'json' and 'yaml'".format(data_format))


def save(transform, filepath, data_format="json", on_not_implemented_error="raise"):
    """
    Take a transform pipeline, serialize it and save a serialized version to a file
    using either json or yaml format.
    Args:
        transform (obj): Transform to serialize.
        filepath (str): Filepath to write to.
        data_format (str): Serialization format. Should be either `json` or 'yaml'.
        on_not_implemented_error (str): Parameter that describes what to do if a transform doesn't implement
            the `to_dict` method. If 'raise' then `NotImplementedError` is raised, if `warn` then the exception will be
            ignored and no transform arguments will be saved.
    """
    check_data_format(data_format)
    transform_dict = to_dict(transform, on_not_implemented_error=on_not_implemented_error)
    dump_fn = json.dump if data_format == "json" else yaml.safe_dump
    with open(filepath, "w") as f:
        dump_fn(transform_dict, f)


def load(filepath, data_format="json", lambda_transforms=None):
    """
    Load a serialized pipeline from a json or yaml file and construct a transform pipeline.
    Args:
        transform (obj): Transform to serialize.
        filepath (str): Filepath to read from.
        data_format (str): Serialization format. Should be either `json` or 'yaml'.
        lambda_transforms (dict): A dictionary that contains lambda transforms, that is instances of the Lambda class.
            This dictionary is required when you are restoring a pipeline that contains lambda transforms. Keys
            in that dictionary should be named same as `name` arguments in respective lambda transforms from
            a serialized pipeline.
    """
    check_data_format(data_format)
    load_fn = json.load if data_format == "json" else yaml.safe_load
    with open(filepath) as f:
        transform_dict = load_fn(f)
    return from_dict(transform_dict, lambda_transforms=lambda_transforms)


def register_additional_transforms():
    """
    Register transforms that are not imported directly into the `albumentations` module.
    """
    try:
        # This import will result in ImportError if `torch` is not installed
        import albumentations.pytorch
    except ImportError:
        pass

@add_metaclass(SerializableMeta)
class BaseCompose:
    def __init__(self, transforms, p):
        self.transforms = Transforms(transforms)
        self.p = p

        self.replay_mode = False
        self.applied_in_replay = False

    def __getitem__(self, item):
        return self.transforms[item]

    def __repr__(self):
        return self.indented_repr()

    def indented_repr(self, indent=REPR_INDENT_STEP):
        args = {k: v for k, v in self._to_dict().items() if not (k.startswith("__") or k == "transforms")}
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += "\n" + " " * (indent - REPR_INDENT_STEP) + "], {args})".format(args=format_args(args))
        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def _to_dict(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],  # skipcq: PYL-W0212
        }

    def get_dict_with_id(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets):
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag, save_key="replay"):
        for t in self.transforms:
            t.set_deterministic(flag, save_key)
class BboxProcessor(DataProcessor):
    @property
    def default_data_name(self):
        return "bboxes"

    def ensure_data_valid(self, data):
        for data_name in self.data_fields:
            data_exists = data_name in data and len(data[data_name])
            if data_exists and len(data[data_name][0]) < 5:
                if self.params.label_fields is None:
                    raise ValueError(
                        "Please specify 'label_fields' in 'bbox_params' or add labels to the end of bbox "
                        "because bboxes must have labels"
                    )
        if self.params.label_fields:
            if not all(i in data.keys() for i in self.params.label_fields):
                raise ValueError("Your 'label_fields' are not valid - them must have same names as params in dict")

    def filter(self, data, rows, cols):
        return filter_bboxes(
            data, rows, cols, min_area=self.params.min_area, min_visibility=self.params.min_visibility
        )

    def check(self, data, rows, cols):
        return check_bboxes(data)

    def convert_from_albumentations(self, data, rows, cols):
        return convert_bboxes_from_albumentations(data, self.params.format, rows, cols, check_validity=True)

    def convert_to_albumentations(self, data, rows, cols):
        return convert_bboxes_to_albumentations(data, self.params.format, rows, cols, check_validity=True)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regrading bounding boxes
    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                params = bbox_params
            else:
                raise ValueError("unknown format of bbox_params, please use `dict` or `BboxParams`")
            self.processors["bboxes"] = BboxProcessor(params, additional_targets)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                params = keypoint_params
            else:
                raise ValueError("unknown format of keypoint_params, please use `dict` or `KeypointParams`")
            self.processors["keypoints"] = KeypointsProcessor(params, additional_targets)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, *args, force_apply=False, **data):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        self._check_args(**data)
        assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = self.transforms if need_to_run else self.transforms.get_always_apply(self.transforms)
        dual_start_end = transforms.start_end if self.processors else None
        check_each_transform = any(
            getattr(item.params, "check_each_transform", False) for item in self.processors.values()
        )

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                for p in self.processors.values():
                    p.preprocess(data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                for p in self.processors.values():
                    p.postprocess(data)
            elif check_each_transform and isinstance(t, DualTransform):
                rows, cols = data["image"].shape[:2]
                for p in self.processors.values():
                    if not getattr(p.params, "check_each_transform", False):
                        continue

                    for data_name in p.data_fields:
                        data[data_name] = p.filter(data[data_name], rows, cols)

        return data

    def _to_dict(self):
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
            }
        )
        return dictionary

    def _check_args(self, **kwargs):
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        # ["bboxes", "keypoints"] could be almost any type, no need to check them
        for data_name, data in kwargs.items():
            internal_data_name = self.additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError("{} must be numpy array type".format(data_name))
            if internal_data_name in checked_multi:
                if data:
                    if not isinstance(data[0], np.ndarray):
                        raise TypeError("{} must be list of numpy arrays".format(data_name))

SERIALIZABLE_REGISTRY = {}





@add_metaclass(SerializableMeta)
class BasicTransform:
    call_backup = None

    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, *args, force_apply=False, **kwargs):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(key in kwargs for key in self.targets_as_params), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag, save_key="replay"):
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(name=self.__class__.__name__, args=format_args(state))

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        if hasattr(self, "mask_fill_value"):
            params["mask_fill_value"] = self.mask_fill_value
        params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'
        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class " + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self):
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self):
        d = self._to_dict()
        d["id"] = id(self)
        return d

class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]

    def apply_to_keypoints(self, keypoints, **params):
        return [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]


class Rotate(DualTransform):
    """Rotate the input by an angle selected randomly from the uniform distribution.
    Args:
        limit ((int, int) or int): range from which a random angle is picked. If limit is a single int
            an angle is picked from (-limit, limit). Default: (-90, 90)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=0.5,
    ):
        super(Rotate, self).__init__(always_apply, p)
        self.limit = to_tuple(limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=345, interpolation=cv2.INTER_LINEAR, **params):
        return F.rotate(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=30, **params):
        return F.rotate(img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def get_params(self):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

    def apply_to_bbox(self, bbox, angle=45, **params):
        return F.bbox_rotate(bbox, angle, params["rows"], params["cols"])

    def apply_to_keypoint(self, keypoint, angle=30, **params):
        return F.keypoint_rotate(keypoint, angle, **params)

    def get_transform_init_args_names(self):
        return ("limit", "interpolation", "border_mode", "value", "mask_value")
class Params:
    def __init__(self, format, label_fields=None):
        self.format = format
        self.label_fields = label_fields

    def _to_dict(self):
        return {"format": self.format, "label_fields": self.label_fields}
class BboxParams(Params):
    """
    Parameters of bounding boxes
    Args:
        format (str): format of bounding boxes. Should be 'coco', 'pascal_voc', 'albumentations' or 'yolo'.
            The `coco` format
                `[x_min, y_min, width, height]`, e.g. [97, 12, 150, 200].
            The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
            The `albumentations` format
                is like `pascal_voc`, but normalized,
                in other words: [x_min, y_min, x_max, y_max]`, e.g. [0.2, 0.3, 0.4, 0.5].
            The `yolo` format
                `[x, y, width, height]`, e.g. [0.1, 0.2, 0.3, 0.4];
                `x`, `y` - normalized bbox center; `width`, `height` - normalized bbox width and height.
        label_fields (list): list of fields that are joined with boxes, e.g labels.
            Should be same type as boxes.
        min_area (float): minimum area of a bounding box. All bounding boxes whose
            visible area in pixels is less than this value will be removed. Default: 0.0.
        min_visibility (float): minimum fraction of area for a bounding box
            to remain this box in list. Default: 0.0.
        check_each_transform (bool): if `True`, then bboxes will be checked after each dual transform.
            Default: `True`
    """

    def __init__(self, format, label_fields=None, min_area=0.0, min_visibility=0.0, check_each_transform=True):
        super(BboxParams, self).__init__(format, label_fields)
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.check_each_transform = check_each_transform

    def _to_dict(self):
        data = super(BboxParams, self)._to_dict()
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "check_each_transform": self.check_each_transform,
            }
        )
        return data


BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


image = cv2.imread('./49269.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rows, cols = image.shape[:2]
bboxes = [[6.48,315.47,269.47,317.4], [90.84,27.53,337.2,474.84]]
category_ids = [18, 19]

# We will use the mapping from category_id to the class name
# to visualize the class label for the bounding box on the image
category_id_to_name = {18: 'dog', 19: 'horse'}
random.seed(7)
transform = Compose(
    [Rotate(p=0.5)],
    bbox_params=BboxParams(format='coco', label_fields=['category_ids']),
)

transformed = transform(image = image,bboxes = bboxes, category_ids= category_ids)
visualize(
    transformed['image'],
    transformed['bboxes'],
    transformed['category_ids'],
    category_id_to_name,)
print(transformed['bboxes'])