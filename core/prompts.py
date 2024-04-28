# from importlib import resources
import os
import functools
import random
import inflect
from importlib import resources

# IE = inflect.engine()
ASSETS_PATH = resources.files("data")

@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)

def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)

def simple_animals():
    return from_file("simple_animals.txt")

def landscape():
    return from_file("landscape.txt", 0, 40)
