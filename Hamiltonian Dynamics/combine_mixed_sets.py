"""
Simple utility script to combine different hetereogeneous sets of a dynamics into
one training file for non meta-learning efforts
"""
import numpy as np
import os

dataset = "Gravity"
subpath = f'{dataset.replace(" ", "_").lower()}3'
print(dataset, subpath)

for setting in ['train.npz', 'test.npz', 'val.npz']:
    combined_images = []
    combined_states = []
    combined_labels = []
    for taskfolder in os.listdir(f"{dataset}/{subpath}/"):
        if '.npz' in taskfolder:
            continue

        data = np.load(f"{dataset}/{subpath}/{taskfolder}/{setting}", allow_pickle=True)
        combined_images.append(data['image'])
        combined_states.append(data['state'])
        combined_labels.append(data['label'])
        
    combined_images = np.vstack(combined_images)
    combined_states = np.vstack(combined_states)
    combined_labels = np.vstack(combined_labels)
    print(combined_images.shape)
    np.savez(f"{dataset}/{subpath}/combined_{setting}", image=combined_images, state=combined_states, label=combined_labels)
    