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
import pygame
import pymunk.pygame_util


class BallBox:
    def __init__(self, dt=0.2, res=(32, 32), init_pos=(3, 3), init_std=0, wall=None, gravity=(0.0, 0.0)):
        pygame.init()

        self.dt = dt
        self.res = res
        if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
            pygame.display.set_mode(res, 0, 24)
            self.screen = pygame.Surface(res, pygame.SRCCOLORKEY, 24)
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, res[0], res[1]), 0)
        else:
            self.screen = pygame.display.set_mode(res, 0, 24)
        self.gravity = gravity
        self.initial_position = init_pos
        self.initial_std = init_std
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = None

        self.dd = 0

    def _clear(self):
        self.screen.fill(pygame.color.THECOLORS["black"])

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        position = np.array(self.initial_position) + self.initial_std * np.random.normal(size=(2,))
        position = np.clip(position, self.dd + radius + 1, self.res[0] - self.dd - radius - 1)
        position = position.tolist()
        body.position = position

        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1.0
        shape.color = pygame.color.THECOLORS["white"]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def run(self, iterations=20, sequences=500, angle_limits=(0, 360), velocity_limits=(10, 25), radius=3,
            flip_gravity=None, save=None, filepath='data/balls.npz', delay=None):
        if save:
            images = np.empty((sequences, iterations, self.res[0], self.res[1]), dtype=np.float32)
            state = np.empty((sequences, iterations, 4), dtype=np.float32)

        dd = self.dd
        self.static_lines = [pymunk.Segment(self.space.static_body, (dd, dd), (dd, self.res[1] - dd), 0.0),
                             pymunk.Segment(self.space.static_body, (dd, dd), (self.res[0] - dd, dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (dd, self.res[1] - dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd, self.res[1] - dd),
                                            (self.res[0] - dd, dd), 0.0)]
        for line in self.static_lines:
            line.elasticity = 1.0
            line.color = pygame.color.THECOLORS["black"]
        # self.space.add(self.static_lines)

        for sl in self.static_lines:
            self.space.add(sl)

        for s in range(sequences):

            if s % 100 == 0:
                print(s)

            angle = np.random.uniform(*angle_limits)
            velocity = np.random.uniform(*velocity_limits)
            # controls[:, s] = np.array([angle, velocity])
            ball = self.fire(angle, velocity, radius)
            for i in range(iterations):
                self._clear()
                self.space.debug_draw(self.draw_options)
                self.space.step(self.dt)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == 'png':
                    pygame.image.save(self.screen, os.path.join(filepath, f"{base_path}s_%02d_%02d.png" % (s, i)))
                elif save == 'npz':
                    images[s, i] = pygame.surfarray.array2d(self.screen).swapaxes(1, 0).astype(np.float32) / (
                                2 ** 24 - 1)
                    state[s, i] = list(ball.body.position) + list(ball.body.velocity)

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)

        return images, state


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
    """ First generate bouncing ball data """
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    scale = 1

    # Get class labels
    train_per_grav = 3000
    val_per_grav = 1000
    test_per_grav = 1000
    num_timesteps = 30
    total_size = train_per_grav + val_per_grav + test_per_grav

    # Velocity limits
    vel_min, vel_max = 5.0, 10.0

    # Create data dir
    base_path = f"mixed_physics/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Sample gravities
    num_gravs = 16
    gs = [[np.cos(k * np.pi / (num_gravs // 2)), np.sin(k * np.pi / (num_gravs // 2))] for k in range(num_gravs)]
    gs = np.array(gs)[[0, 4, 8, 12], :]

    # Set up files
    np.random.seed(321)

    # Over each grav, sample
    for idx, g in enumerate(gs):
        g_range = 2.5 # + np.random.random_sample()
        g_x, g_y = np.array(g) * g_range
        print(f"=> Idx {idx}, Gravity Range: {g_range}")
        print(f"=> Gx {g_x} Gy {g_y}")
        cannon = BallBox(dt=0.25, res=(32 * scale, 32 * scale), init_pos=(16 * scale, 16 * scale), init_std=8, wall=None, gravity=(g_x, g_y))
        i, s = cannon.run(delay=None, iterations=num_timesteps, sequences=total_size + 2000, radius=3 * scale, angle_limits=(0, 360), velocity_limits=(vel_min, vel_max), save='npz')

        i = (i > 0).astype(float)

        # Brute force check for any bad trajectories where the ball leaves
        bad_indices = []
        for seq_idx, sequence in enumerate(i):
            sums = np.sum(sequence, axis=(1, 2))
            if np.where(sums == 0.0)[0].shape[0] > 0 or np.where(sums > (32 * 5))[0].shape[0] > 0:
                bad_indices.append(seq_idx)

        i = np.delete(i, bad_indices, 0)
        s = np.delete(s, bad_indices, 0)
        if i.shape[0] > total_size:
            i = i[:total_size, :]
            s = s[:total_size, :]

        # Shuffle the set
        p = np.random.permutation(total_size)
        i = i[p]
        s = s[p]

        # Break into train and test sets, adding in generic labels
        train_images = i[:train_per_grav]
        train_states = s[:train_per_grav]
        train_classes = np.full([train_images.shape[0], 1], fill_value=idx)

        val_images = i[train_per_grav:train_per_grav + val_per_grav]
        val_states = s[train_per_grav:train_per_grav + val_per_grav]
        val_classes = np.full([val_images.shape[0], 1], fill_value=idx)

        test_images = i[train_per_grav + val_per_grav:]
        test_states = s[train_per_grav + val_per_grav:]
        test_classes = np.full([test_images.shape[0], 1], fill_value=idx)

        print(f"Train - Images: {train_images.shape} | States: {train_states.shape} | Classes: {train_classes.shape}")
        print(f"Val - Images: {val_images.shape} | States: {val_states.shape} | Classes: {val_classes.shape}")
        print(f"Test - Images: {test_images.shape} | States: {test_states.shape} | Classes: {test_classes.shape}")

        # Save the individual gravity to its own data paths
        if not os.path.exists(f"{base_path}/mixed_{idx}/"):
            os.mkdir(f"{base_path}/mixed_{idx}/")

        # Save sets
        np.savez(os.path.abspath(f"{base_path}/mixed_{idx}/train.npz"), image=train_images, state=train_states, label=train_classes)
        np.savez(os.path.abspath(f"{base_path}/mixed_{idx}/val.npz"), image=val_images, state=val_states, label=val_classes)
        np.savez(os.path.abspath(f"{base_path}/mixed_{idx}/test.npz"), image=test_images, state=test_states, label=test_classes)

    """ Then generate all DM hamiltonian data """
    # Dataset parameters
    print("=> Generating datasets....")
    for dataset, class_id in zip(
        ["pendulum_00", "pendulum_05", "pendulum_10", "pendulum_15", 
         "mass_spring_00", "mass_spring_05", "mass_spring_10", "mass_spring_15"], 
        [4, 5, 6, 7, 8, 9, 10, 11]
    ):
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
            folder='mixed_physics/',
            dataset=dataset,
            dt=dt,
            num_steps=num_steps,
            steps_per_dt=steps_per_dt,
            num_train=num_total,
            num_test=num_test,
            overwrite=overwrite,
        )

        """ Data Generation """
        dataset_path = f'mixed_physics/{dataset}/'
        
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
        labels = np.full([images.shape[0], 1], fill_value=class_id)
        print(images.shape)

        # Bernoullize the dynamics to foreground and background
        images = (images > 0.45).astype(float)

        # Split and save into groups
        if not os.path.exists(f"mixed_physics/mixed_{class_id}/"):
            os.mkdir(f"mixed_physics/mixed_{class_id}/")
            
        np.savez(f"mixed_physics/mixed_{class_id}/train.npz", image=images[:num_train], state=states[:num_train], label=labels[:num_train])
        np.savez(f"mixed_physics/mixed_{class_id}/val.npz", image=images[num_train:num_train + num_val], state=states[num_train:num_train + num_val], label=labels[num_train:num_train + num_val])
        np.savez(f"mixed_physics/mixed_{class_id}/test.npz", image=images[num_train + num_val:], state=states[num_train + num_val:], label=labels[num_train + num_val:])
        shutil.rmtree(f"mixed_physics/{dataset}/")

        print(images[:num_train].shape, images[num_train:num_train + num_val].shape, images[num_train + num_val:].shape)
        print(states[:num_train].shape, states[num_train:num_train + num_val].shape, states[num_train + num_val:].shape)
        print(labels[:num_train].shape, labels[num_train:num_train + num_val].shape, labels[num_train + num_val:].shape)
