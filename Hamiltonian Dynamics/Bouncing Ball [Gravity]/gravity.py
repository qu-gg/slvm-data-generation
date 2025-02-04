import pygame
import pymunk.pygame_util
import numpy as np
import os


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


if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    scale = 1

    # Get class labels
    train_per_grav = 10000
    val_per_grav = 2000
    test_per_grav = 2000
    num_timesteps = 20
    total_size = train_per_grav + val_per_grav + test_per_grav

    # Velocity limits
    vel_min, vel_max = 5.0, 10.0

    # Create data dir
    base_path = f"gravity/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Set up files
    np.random.seed(321)

    # Over each grav, sample
    cannon = BallBox(dt=0.25, res=(32 * scale, 32 * scale), init_pos=(16 * scale, 16 * scale), init_std=8, wall=None, gravity=(0, 2.5))
    i, s = cannon.run(delay=None, iterations=num_timesteps, sequences=total_size + 2000, radius=3 * scale, angle_limits=(45, 135), velocity_limits=(vel_min, vel_max), save='npz')
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
    train_classes = np.full([train_images.shape[0], 1], fill_value=0)

    val_images = i[train_per_grav:train_per_grav + val_per_grav]
    val_states = s[train_per_grav:train_per_grav + val_per_grav]
    val_classes = np.full([val_images.shape[0], 1], fill_value=0)

    test_images = i[train_per_grav + val_per_grav:]
    test_states = s[train_per_grav + val_per_grav:]
    test_classes = np.full([test_images.shape[0], 1], fill_value=0)

    print(f"Train - Images: {train_images.shape} | States: {train_states.shape} | Classes: {train_classes.shape}")
    print(f"Val - Images: {val_images.shape} | States: {val_states.shape} | Classes: {val_classes.shape}")
    print(f"Test - Images: {test_images.shape} | States: {test_states.shape} | Classes: {test_classes.shape}")

    # Save sets
    np.savez(os.path.abspath(f"gravity/train.npz"), image=train_images, state=train_states, label=train_classes)
    np.savez(os.path.abspath(f"gravity/val.npz"), image=val_images, state=val_states, label=val_classes)
    np.savez(os.path.abspath(f"gravity/test.npz"), image=test_images, state=test_states, label=test_classes)
