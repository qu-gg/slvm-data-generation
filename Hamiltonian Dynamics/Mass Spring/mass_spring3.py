"""
@file generate_date.py
@url https://github.com/deepmind/dm_hamiltonian_dynamics/tree/master/
"""
from matplotlib import pyplot as plt
from matplotlib import animation as plt_animation
import numpy as np
from jax import config as jax_config
import shutil

from tqdm import tqdm
import os

jax_config.update("jax_enable_x64", True)

from simulation_lib import load_datasets
from simulation_lib import datasets


def unstack(value: np.ndarray, axis: int = 0):
    """Unstacks an array along an axis into a list"""
    split = np.split(value, value.shape[axis], axis=axis)
    return [np.squeeze(v, axis=axis) for v in split]


def make_batch_grid(
        batch: np.ndarray,
        grid_height: int,
        grid_width: int,
        with_padding: bool = True):
    """Makes a single grid image from a batch of multiple images."""
    assert batch.ndim == 5
    assert grid_height * grid_width >= batch.shape[0]
    batch = batch[:grid_height * grid_width]
    batch = batch.reshape((grid_height, grid_width) + batch.shape[1:])
    if with_padding:
        batch = np.pad(batch, pad_width=[[0, 0], [0, 0], [0, 0],
                                         [1, 0], [1, 0], [0, 0]],
                       mode="constant", constant_values=1.0)
    batch = np.concatenate(unstack(batch), axis=-3)
    batch = np.concatenate(unstack(batch), axis=-2)
    if with_padding:
        batch = batch[:, 1:, 1:]
    return batch


def plot_animattion_from_batch(
        batch: np.ndarray,
        grid_height,
        grid_width,
        with_padding=True,
        figsize=None):
    """Plots an animation of the batch of sequences."""
    if figsize is None:
        figsize = (grid_width, grid_height)
    batch = make_batch_grid(batch, grid_height, grid_width, with_padding)
    batch = batch[:, ::-1]
    fig = plt.figure(figsize=figsize)
    plt.close()
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')
    img = ax.imshow(batch[0])

    def frame_update(i):
        i = int(np.floor(i).astype("int64"))
        img.set_data(batch[i])
        return [img]

    anim = plt_animation.FuncAnimation(
        fig=fig,
        func=frame_update,
        frames=np.linspace(0.0, len(batch), len(batch) * 5 + 1)[:-1],
        save_count=len(batch),
        interval=10,
        blit=True
    )
    return anim


def plot_sequence_from_batch(
        batch: np.ndarray,
        t_start: int = 0,
        with_padding: bool = True,
        fontsize: int = 20):
    """Plots all of the sequences in the batch."""
    n, t, dx, dy = batch.shape[:-1]
    xticks = np.linspace(dx // 2, t * (dx + 1) - 1 - dx // 2, t)
    xtick_labels = np.arange(t) + t_start
    yticks = np.linspace(dy // 2, n * (dy + 1) - 1 - dy // 2, n)
    ytick_labels = np.arange(n)
    batch = batch.reshape((n * t, 1) + batch.shape[2:])
    batch = make_batch_grid(batch, n, t, with_padding)[0]
    plt.imshow(batch.squeeze())
    plt.xticks(ticks=xticks, labels=xtick_labels, fontsize=fontsize)
    plt.yticks(ticks=yticks, labels=ytick_labels, fontsize=fontsize)


def visalize_dataset(
        dataset_path: str,
        sequence_lengths: int = 60,
        grid_height: int = 2,
        grid_width: int = 5):
    """Visualizes a dataset loaded from the path provided."""
    split = "test"
    batch_size = grid_height * grid_width
    dataset = load_datasets.load_dataset(
        path=dataset_path,
        tfrecord_prefix=split,
        sub_sample_length=sequence_lengths,
        per_device_batch_size=batch_size,
        num_epochs=None,
        drop_remainder=True,
        shuffle=False,
        shuffle_buffer=100
    )
    sample = next(iter(dataset))
    batch_x = sample['x'].numpy()
    batch_image = sample['image'].numpy()
    # Plot real system dimensions
    plt.figure(figsize=(24, 8))
    for i in range(batch_x.shape[-1]):
        plt.subplot(1, batch_x.shape[-1], i + 1)
        plt.title(f"Samples from dimension {i + 1}")
        plt.plot(batch_x[:, :, i].T)
    plt.show()
    # Plot a sequence of 50 images
    plt.figure(figsize=(30, 10))
    plt.title("Samples from 50 steps sub sequences.")
    plot_sequence_from_batch(batch_image[:, :50])
    plt.show()
    # Plot animation
    return plot_animattion_from_batch(batch_image, grid_height, grid_width)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mass_spring')
    parser.add_argument('--class_id', type=int, default=0)
    args = parser.parse_args()

    # Dataset parameters
    print("=> Generating datasets....")
    for dataset, class_id in zip(["mass_spring_2g", "mass_spring_3g", "mass_spring_4g"], [0, 1, 2]):
        num_steps = 30
        dt = 0.1
        steps_per_dt = 1

        num_train = 3000
        num_val = 1000
        num_test = 1000
        num_total = num_train + num_val + num_test

        # Generate dataset
        overwrite = True
        datasets.generate_full_dataset(
            folder='mass_spring3/',
            dataset=dataset,
            dt=dt,
            num_steps=num_steps,
            steps_per_dt=steps_per_dt,
            num_train=num_total,
            num_test=num_test,
            overwrite=overwrite,
        )

        """ Data Generation """
        dataset_path = f'mass_spring3/{dataset}/'
        
        print("=> Converting training files...")
        loaded_dataset = load_datasets.load_dataset(
            path=dataset_path,
            tfrecord_prefix="train",
            sub_sample_length=num_steps,
            per_device_batch_size=1,
            num_epochs=1,
            drop_remainder=False
        )

        print(loaded_dataset)

        images = []
        states = []
        for idx, sample in tqdm(enumerate(loaded_dataset)):
            image = sample['image'].numpy()
            print(image.shape)

            # (32, 32, 3) -> (3, 32, 32)
            image = np.swapaxes(image, 3, 4)
            image = np.swapaxes(image, 2, 3)

            # Append to stack
            images.append(image)
            states.append(sample['x'].numpy())

        images = np.vstack(images)
        states = np.vstack(states)
        labels = np.full([images.shape[0], 1], fill_value=args.class_id)
        print(images.shape)

        # Bernoullize the dynamics to foreground and background
        images = (images > 0.45).astype(float)

        # Split and save into groups
        if not os.path.exists(f"mass_spring3/mass_spring_{class_id}/"):
            os.mkdir(f"mass_spring3/mass_spring_{class_id}/")
            
        np.savez(f"mass_spring3/mass_spring_{class_id}/train.npz", image=images[:num_train], state=states[:num_train], label=labels[:num_train])
        np.savez(f"mass_spring3/mass_spring_{class_id}/val.npz", image=images[num_train:num_train + num_val], state=states[num_train:num_train + num_val], label=labels[num_train:num_train + num_val])
        np.savez(f"mass_spring3/mass_spring_{class_id}/test.npz", image=images[num_train + num_val:], state=states[num_train + num_val:], label=labels[num_train + num_val:])
        shutil.rmtree(f"mass_spring3/{dataset}/")

        print(images[:num_train].shape, images[num_train:num_train + num_val].shape, images[num_train + num_val:].shape)
        print(states[:num_train].shape, states[num_train:num_train + num_val].shape, states[num_train + num_val:].shape)
        print(labels[:num_train].shape, labels[num_train:num_train + num_val].shape, labels[num_train + num_val:].shape)
