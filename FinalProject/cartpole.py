"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

logger = logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
#------------variables added for alex_si physics-------------------------------
        self.friction_cart=5.e-4
        self.friction_pend=2.e-6
        self.phi = 0;
#------------variables added for actuator noise---------------------------
        random.seed(a=67)
        # uncomment the appropriate actuator noise model
        self.act_noise=0
        #self.act_noise=self.force_mag*0.05*random.uniform(-0.5,0.5) # 5% uniform actuator noise
        #self.act_noise=self.force_mag*0.10*random.uniform(-0.5,0.5) # 10% uniform actuator noise
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag+self.act_noise if action==1 else -self.force_mag-self.act_noise
#------------------original physics----------------------------------------------------------------------------------------------------------------
#        costheta = math.cos(theta)
#        sintheta = math.sin(theta)
#        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
#        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
#        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
#        x  = x + self.tau * x_dot
#        x_dot = x_dot + self.tau * xacc
#        theta = theta + self.tau * theta_dot
#        theta_dot = theta_dot + self.tau * thetaacc

#-----------------alex_si physics try1------------------------------------------------------------
#Tried to implement Si's implementation of sutton equation. Issues is that there is no way to factor time step whcich is required for this model to work
#        #using si's variable names
#        angle=theta
#        angular_vel=theta_dot
#        distance=x
#        vel=x_dot
#        friction_cart=self.friction_cart
#        friction_pend=self.friction_pend
#        length=self.length
#        m_pend=self.masspole
#        phi=self.phi
#        g=self.gravity
#        total_m=self.total_mass
#        momt_pend=self.polemass_length
#        
#        #following lines of code are direct from si (unless specified otherwise)
#        hforce=momt_pend*(angular_vel*angular_vel)*math.sin(angle);
#        part_num=-force-hforce+friction_cart*np.sign(vel);
#
#        denom_ang_vel=length*(4/3-m_pend*((math.cos(angle))*(math.cos(angle)))/total_m);
#        num_ang_vel=g*math.sin(angle)*math.cos(phi)+math.cos(angle)*part_num/total_m-friction_pend*angular_vel/momt_pend;
#
#        theta_dot=angular_vel; #writing si variable back to correct variable
#        theta=num_ang_vel/denom_ang_vel; #writing si variable back to correct variable
#        dxdt2=num_ang_vel/denom_ang_vel;
#
#        num_vel=force-total_m*g*math.sin(phi)+hforce-momt_pend*dxdt2*math.cos(angle)-friction_cart*np.sign(vel);
#
#        x_dot=vel; #writing si variable back to correct variable
#        x=num_vel/total_m; #writing si variable back to correct variable
#-----------------end alex_si physics try1--------------------------------------------------------

# All I did for this one was add the friction terms as described in eq 33 & 34 in Si, Wang paper
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta-self.friction_cart * np.sign(x_dot)) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp - ((self.friction_pend * theta_dot)/self.polemass_length)) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
#-----------------end alex_si physics try2--------------------------------------------------------

        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
