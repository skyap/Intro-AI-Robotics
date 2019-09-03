#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import sys

class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass

import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
from skimage.io import imsave
import datetime
from skimage.transform import resize
import tqdm


with add_path('..'):
    gym_duckietown = __import__('gym_duckietown')

# import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown')
parser.add_argument('--map-name', default='loop_empty')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed = args.seed,
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion
    )
else:
    env = gym.make(args.env_name)

env.reset()
img = env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


counter = 0
n=0
filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
imgs = []
actions = []
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global counter
    global filename
    global imgs
    global n
    global actions
    global img

    action = np.array([0.0, 0.0])
    
    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
        if env.unwrapped.step_count%10==0:
            imgs.append(img)
            actions.append(0)
            counter+=1


    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
        imgs.append(img)
        actions.append(3)
        counter+=1
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
        imgs.append(img)
        actions.append(1)
        counter+=1
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
        imgs.append(img)
        actions.append(2)
        counter+=1
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
        print("save images to folder")
        for i in range(len(imgs)):
            imsave(f'./data/{filename}_{n+i:05d}_{actions[i]}.png', resize(imgs[i],(120,160,3),preserve_range=True).astype(np.uint8))
        print("Done Saving")
        imgs = []
        actions = []
        n = counter


    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5
    # action = expert.predict(None)
    obs, reward, done, info = env.step(action)
    # print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))
    # print(expert.predict(None))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    img = env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()