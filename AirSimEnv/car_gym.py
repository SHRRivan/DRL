import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from AirSimEnv.airsim_env import AirSimEnv
from typing import Tuple, Optional


class CarGym(AirSimEnv):
    def __init__(
            self,
            ip_address: Optional[str] = '127.0.0.1',
            image_shape: Tuple = (84, 84, 1),
            max_speed: int = 300,
            min_speed: int = 10,
            thresh_dist: float = 3.5,
            beta: int = 3,
            api_cont: bool = True,
            _init_env: bool = True,
    ):
        super().__init__(image_shape)
        self.ip_address = ip_address
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.thresh_dist = thresh_dist
        self.beta = beta
        self.api_cont = api_cont
        self.car, self.state, self.image_request = None, None, None
        self.car_cont, self.car_state = None, None
        self.ep_len, self.ep_rew = 0, 0
        if _init_env:
            self.init_car()

    def init_car(self):
        # initial the state dictionary for car state
        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        # build connection through python api
        # the default address is None, and str like "127.0.0.1" is optional
        try:
            self.car = airsim.CarClient(ip=self.ip_address)
        except TypeError:
            self.car = airsim.CarClient()

        self.car.confirmConnection()
        self.car.enableApiControl(self.api_cont)

        self.car_cont = airsim.CarControls()
        # create an image request from camera "0", image type is depth perspective,
        # pixels as float = True, compress= False
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(self.api_cont)
        # True--unlock the simulation car, means it can move around
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action: np.ndarray):
        """
        transform action into carControls and feeds it into the airsim client(). Then it goes to sleep.
        :param action: a 3-element array, where each element increases/decreases the corresponding carControl() object
        :return: None
        """
        # throttle  0.0
        new_throttle = self.car_cont.throttle + float(action[0])
        if new_throttle >= 1.0:
            new_throttle = 1.0
        elif new_throttle < -1.0:
            new_throttle = -1.0
        self.car_cont.throttle = new_throttle

        if float(action[1]) > 0.5:
            self.car_cont.brake = 1
        else:
            self.car_cont.brake = 0

        # steering = 0.0
        new_steering = self.car_cont.steering + float(action[2])
        if new_steering >= 1.0:
            new_steering = 1.0
        elif new_steering <= -1.0:
            new_steering = -1.0
        self.car_cont.steering = new_steering

        self.car.setCarControls(self.car_cont)
        # time.sleep(1)      # I guess when the policy output delta, no need to do time-sleep

    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        # return im_final.reshape([1, 84, 84])
        return im_final.reshape(self.shape)

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()
        collision = self.car.simGetCollisionInfo().has_collided

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = collision

        return image

    def _compute_reward(self):
        """
        :param: None
        :return: reward, done
        """
        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        if dist > self.thresh_dist:
            reward = -3
        else:
            reward_dist = math.exp(-self.beta * dist) - 0.5
            reward_speed = (
                                   (self.car_state.speed - self.min_speed) / (self.max_speed - self.min_speed)
                           ) - 0.5
            reward = reward_dist + reward_speed

        done = False
        if reward < -1:
            done = True
        # if self.car_cont.brake == 0:
        #     if self.car_state.speed <= 1:      # change <= 1 to <= self.min_speed
        #         done = True
        if self.state["collision"]:
            done = True

        return reward, done

    def step(self, action: np.ndarray):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        self.ep_len += 1
        self.ep_rew += reward

        return obs, reward, done, self.state

    def reset(self):
        self.ep_len, self.ep_rew = 0, 0
        self._setup_car()
        self._do_action(np.array([0., 0., 0.]))
        return self._get_obs()

    def get_episode_info(self):
        return self.ep_len, self.ep_rew


if __name__ == '__main__':
    env = CarGym()
    # _ = env.reset()
    # action = np.array([0.211, 0, 0])
    # while True:
    #     _, _, _, _ = env.step(action)
