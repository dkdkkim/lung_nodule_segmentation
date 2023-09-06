import numpy as np
from scipy.ndimage import rotate
import random

def _normalize_uint8(inp: np.ndarray):
    output = inp / 255.
    return output

def _unsqueeze(inp: np.ndarray):
    output = np.expand_dims(inp, axis=0)
    return output

def _center_crop(inp: np.ndarray,
                crop_size: list):
    shp = inp.shape
    output = inp[int(shp[0]/2 - crop_size[0]/2):int(shp[0]/2 + crop_size[0]/2),
             int(shp[1]/2 - crop_size[1]/2):int(shp[1]/2 + crop_size[1]/2),
             int(shp[2]/2 - crop_size[2]/2):int(shp[2]/2 + crop_size[2]/2)]
    return output

def _shift_crop(inp: np.ndarray,
                crop_size: list,
                shift: list)-> np.ndarray:
    shp = inp.shape
    output = inp[int(shp[0]/2 - crop_size[0]/2 + shift[0]):int(shp[0]/2 + crop_size[0]/2 + shift[0]),
             int(shp[1]/2 - crop_size[1]/2 + shift[1]):int(shp[1]/2 + crop_size[1]/2 + shift[1]),
             int(shp[2]/2 - crop_size[2]/2 + shift[2]):int(shp[2]/2 + crop_size[2]/2 + shift[2])]
    return output

def _rotate_xaxis(inp: np.ndarray,
                angle: int) -> np.ndarray:
    return rotate(inp, angle, axes=(0, 1), reshape=False)

def _rotate_yaxis(inp: np.ndarray,
                angle: int) -> np.ndarray:
    return rotate(inp, angle, axes=(0, 2), reshape=False)

def _rotate_zaxis(inp: np.ndarray,
                angle: int) -> np.ndarray:
    return rotate(inp, angle, axes=(1, 2), reshape=False)

def _flip(inp: np.ndarray,
                flip_mode: int):
    if flip_mode == 0:
        return inp
    elif flip_mode == 1:
        return np.flip(inp, 0)
    elif flip_mode == 2:
        return np.flip(inp, 2)
    elif flip_mode == 3:
        return np.flip(np.flip(inp, 0), 2)

class Flip:
    """ Flip image on coronal plane """
    def __init__(self):
        pass
    
    def __call__(self, image, mask):
        flip_mode = random.sample([0, 1, 2, 3], 1)[0]
        flip_img = _flip(image, flip_mode)
        flip_mask = _flip(mask, flip_mode)
        return flip_img, flip_mask
    
    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Rotate:
    """ Rotate images on y axis """

    def __init__(self,
                angles=[0, 90, 180, 270]):
        self.angles = angles
    
    def __call__(self, inp, mask):
        angle_x = random.sample(self.angles, 1)[0]
        angle_y = random.sample(self.angles, 1)[0]
        angle_z = random.sample(self.angles, 1)[0]
        # x-rotation
        rot_img = _rotate_xaxis(inp, angle_x)
        rot_mask = _rotate_xaxis(mask, angle_x)
        #y-rotation
        rot_img = _rotate_yaxis(rot_img, angle_y)
        rot_mask = _rotate_yaxis(rot_mask, angle_y)
        #z-rotation
        rot_img = _rotate_zaxis(rot_img, angle_z)
        rot_mask = _rotate_zaxis(rot_mask, angle_z)
        
        return rot_img, rot_mask
    
    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize:
    """Normalize uint type images"""

    def __init__(self):
        pass
    
    def __call__(self, inp, mask):
        output = _normalize_uint8(inp)
        return output, mask

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Unsqueeze:
    """Unsqueeze uint type images"""

    def __init__(self):
        pass

    def __call__(self, inp, mask):
        inp = _unsqueeze(inp)
        return inp, mask

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class CenterCrop:
    """Crop image on center position"""

    def __init__(self,
              crop_size:list=[48,24,48],
              ):
        self.cs = crop_size

    def __call__(self, image, mask):
        image = _center_crop(image, self.cs)
        mask = _center_crop(mask, self.cs)
        return image, mask

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomCrop:
    """Crop image on ramdom position near center"""

    def __init__(self,
              crop_size:list=[48,24,48],
              shift_ratio=0.05):
        self.cs = crop_size
        self.shift_ratio = shift_ratio

    def __call__(self, image, mask):
        shift = [random.sample(list(range(-int(self.cs[0]*self.shift_ratio),int(self.cs[0]*self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[1]*self.shift_ratio),int(self.cs[1]*self.shift_ratio)+1)),1)[0],
                random.sample(list(range(-int(self.cs[2]*self.shift_ratio),int(self.cs[2]*self.shift_ratio)+1)),1)[0]]
        image = _shift_crop(image, self.cs, shift)
        mask = _shift_crop(mask, self.cs, shift)
        return image, mask

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Compose:
    """Compose several transforms together"""
    def __init__(self,
                 transforms: list
                 ):
        self.transforms = transforms
    def __call__(self, inp, mask):
        # print (f"compose: {inp.shape}, {mask.shape}")
        for t in self.transforms:
            inp, mask = t(inp, mask)
            # print(f"{t.__repr__()}, {inp.shape}, {mask.shape}")
        return inp, mask
    def __repr__(self): return str([transform for transform in self.transforms])

