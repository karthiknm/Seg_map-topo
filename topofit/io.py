import itertools
import numpy as np
import surfa as sf
import torch

from . import ico
from . import utils

# image shape used during training
target_image_shape = (36, 36, 36)  # Updated to match the generated file size (36x36x36), Change if you want to use different resolutions.

def load_subject_data(subj, hemi, ground_truth=False, low_res=False):
    """
    Load a subject's one-hot encoded segmentation map and surface mesh.
    """
    seg_path = f'{subj}/seg.npy'  

    # Load the one-hot encoded segmentation map
    seg_data = np.load(seg_path)

    # Ensure the segmentation map has the correct shape: (C, H, W, D)

    seg_data = np.transpose(seg_data, (3, 0, 1, 2))  # Now seg_data is (5, 36, 36, 36)


    # Convert the segmentation map to a torch tensor
    input_image = torch.from_numpy(seg_data).float()

    # Load the initial template surface mesh
    template = ico.get_initial_template(hemi)
    input_vertices = template.vertices.astype(np.float32)
    input_vertices = input_vertices[ico.get_mapping(6, 1)]


    data = {
        'input_image': input_image,
        'input_vertices': torch.from_numpy(input_vertices),
    }

    if ground_truth:
        s = subj[-19:]  # THIS IS DONE TO GET SUBJECT NAME ONLY, DO NOT CHANGE
        true_vertices = sf.load_mesh(f'/local/scratch/v_karthik_mohan/gt/{s}/{hemi}.white.surf')
        true_vertices = true_vertices.convert(space='vox').vertices.astype(np.float32)
        if low_res:
            true_vertices = true_vertices[ico.get_mapping(7, 6)]
        data['true_vertices'] = torch.from_numpy(true_vertices)

    return data

def compute_image_cropping(image_shape, vertices):
    """
    Compute the correct image cropping given the bounding box of aligned vertices
    """
    vmin = vertices.min(0)
    vmax = vertices.max(0)

    image_limit = np.asarray(image_shape) - 1
    pmin = np.clip(np.floor(vertices.min(0)), (0, 0, 0), image_limit)
    pmax = np.clip(np.ceil(vertices.max(0)),  (0, 0, 0), image_limit)

    pdiff = np.asarray(target_image_shape) - (pmax - pmin)
    if np.any(pdiff < 0):
        raise RuntimeError('alignment exceeds target image shape')

    # pad is necessary
    padding = pdiff / 2.0
    pmin = np.clip(pmin - np.floor(padding), (0, 0, 0), image_limit)
    pmax = np.clip(pmax + np.ceil(padding), (0, 0, 0), image_limit)
    source_shape = pmax - pmin
    cropping = tuple([slice(int(a), int(b)) for a, b in zip(pmin, pmax)])
    return cropping


class InfiniteSampler(torch.utils.data.IterableDataset):
    """
    Iterable torch dataset that infinitely samples training subjects.
    """
    def __init__(self, hemi, training_subjs, low_res):
        super().__init__()
        self.hemi = hemi
        self.training_subjs = training_subjs
        self.low_res = low_res

    def __iter__(self):
        yield from itertools.islice(self.infinite(), 0, None, 1)

    def infinite(self):
        while True:
            idx = np.random.randint(len(self.training_subjs))
            subj = self.training_subjs[idx]
            try:
                data = load_subject_data(subj, self.hemi, ground_truth=True, low_res=self.low_res)
                data = {k: v for k, v in data.items() if k in ('input_image', 'input_vertices', 'true_vertices')}
            except RuntimeError:
                continue
            yield from [data]

class Collator:
    def __init__(self, data):
        self.data = data[0]

    def pin_memory(self):
        for key, value in self.data.items():
            self.data[key] = value.pin_memory()
        return self

def get_data_loader(hemi, training_subjs, low_res=False, prefetch_factor=8):
    collate_fn = lambda batch: Collator(batch)
    sampler = InfiniteSampler(hemi, training_subjs, low_res)
    data_loader = torch.utils.data.DataLoader(sampler, batch_size=1, num_workers=1,
        prefetch_factor=prefetch_factor, collate_fn=collate_fn, pin_memory=True)
    return data_loader
