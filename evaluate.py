#!/usr/bin/env python

"""
Script to evaluate a custom trained TopoFit model. If this code is
useful to you, please cite:

TopoFit: Rapid Reconstruction of Topologically-Correct Cortical Surfaces
Andrew Hoopes, Juan Eugenio Iglesias, Bruce Fischl, Douglas Greve, Adrian Dalca
Medical Imaging with Deep Learning. 2022.
"""

import os
import argparse
import numpy as np
import surfa as sf
import torch
import topofit


parser = argparse.ArgumentParser()
parser.add_argument('--subjs', nargs='+', required=True, help='subject(s) to evaluate')
parser.add_argument('--hemi', required=True, help='hemisphere to evaluate (`lr` or `rh`)')
parser.add_argument('--model', required=True, help='model file (.pt) to load')
parser.add_argument('--suffix', default='topofit', help='custom ')
parser.add_argument('--gpu', default='0', help='GPU device ID (default is 0')
parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
parser.add_argument('--xhemi', action='store_true', help='Xhemi')
args = parser.parse_args()

print(f'Xhemi {args.xhemi}');

# sanity check on inputs
if args.hemi not in ('lh', 'rh'):
    print("error: hemi must be 'lh' or 'rh'")
    exit(1)

# configure device
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cpu')
else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda')
topofit.utils.set_device(device)

# configure model
print('Configuring model')
model = topofit.model.SurfNet().to(device)

# initialize model weights
print(f'Loading model weights from {args.model}')
weights = torch.load(args.model, map_location=device)
model.load_state_dict(weights['model_state_dict'])

# enable evaluation mode
model.train(mode=False)

# start training loop
for subj in args.subjs:
    
    # load subject data
    data = topofit.io.load_subject_data(subj, args.hemi, ground_truth = False)

    # predict surface
    with torch.no_grad():
        input_image = data['input_image'].to(device)
        input_vertices = data['input_vertices'].to(device)
        result, topology = model(input_image, input_vertices)
        vertices = result['pred_vertices'].cpu().numpy()
        faces = topology['faces'].cpu().numpy()

    # build mesh and convert to correct space and geometry
    surf = sf.Mesh(vertices, faces, space='vox')
    surf = surf.convert(geometry=data['input_geometry'])

    # write surface
    filename = os.path.join(subj, 'surf', f'{args.hemi}.white.{args.suffix}')
    surf.save(filename)
    print(f'Saved white-matter surface to {filename}')
