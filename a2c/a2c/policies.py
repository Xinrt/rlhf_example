import numpy as np
import tensorflow as tf
from a2c.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample
import gym

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):         
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, nlstm=256, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):           
            a, v, s = sess.run([a0, v0, snew], {X:ob, S:state, M:mask})
            return a, v, s

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x)
            vf = fc(h4, 'v', 1, act=lambda x:x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):         
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class MlpPolicy(object):

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 nenv,
                 nsteps,
                 nstack,
                 reuse=False):
        nbatch = nenv*nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            x = tf.cast(X, tf.float32)/255.

            # Only look at the most recent frame
            x = x[:, :, :, -1]

            w, h = x.get_shape()[1:]
            x = tf.reshape(x, [-1, int(w * h)])
            x = fc(x, 'fc1', nh=2048, init_scale=np.sqrt(2))
            x = fc(x, 'fc2', nh=1024, init_scale=np.sqrt(2))
            x = fc(x, 'fc3', nh=512,  init_scale=np.sqrt(2))
            pi = fc(x, 'pi', nact, act=lambda x: x)
            vf = fc(x, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        

class MlpContPolicy(object):

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 nenv,
                 nsteps,
                 nstack,
                 reuse=False):
        

        nbatch = nenv*nsteps


        # if len(ob_space.shape) == 3:
        #     nh, nw, nc = ob_space.shape  # If 3D observation space (like image)
        # elif len(ob_space.shape) == 1:
        #     nh = ob_space.shape[0]  # If 1D observation vector
        #     nw, nc = 1, 1  # Set default values as these dimensions do not exist
        # else:
        #     raise ValueError(f"Unexpected ob_space.shape: {ob_space.shape}")
        # # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc*nstack)

        # if isinstance(ac_space, gym.spaces.Box):
        #     nact = ac_space.shape[0]  # Obtain the number of actions for a continuous action space
        # # Discrete
        # elif isinstance(ac_space, gym.spaces.Discrete):
        #     nact = ac_space.n
        # else:
        #     raise ValueError(f"Unsupported action space type: {type(ac_space)}")

        ob_shape = (nbatch, ob_space.shape[0] * nstack)
        
        nact = ac_space.shape[0] if isinstance(ac_space, gym.spaces.Box) else ac_space.n
        # nact = ac_space.n
        # X = tf.placeholder(tf.uint8, ob_shape)  # obs
        X = tf.placeholder(tf.float32, ob_shape, name="Ob")
        with tf.variable_scope("model", reuse=reuse):
            # x = tf.cast(X, tf.float32)/255.
            x = X  # No need to cast or reshape

            # Only look at the most recent frame
            # x = x[:, :, :, -1]

            # w, h = x.get_shape()[1:]
            x = fc(x, 'fc1', nh=64, init_scale=np.sqrt(2))
            x = tf.nn.tanh(x)
            x = fc(x, 'fc2', nh=64, init_scale=np.sqrt(2))
            x = tf.nn.tanh(x)
            pi = fc(x, 'pi', nact, act=lambda x: x)
            vf = fc(x, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = pi  # For continuous action spaces, we don't need to sample
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
