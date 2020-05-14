#!/usr/bin/env python3

# This is the predictive model we are building
# P(L | H, E) = P(H | L, E)P(L | E) # MLE
# P(theta | H, E, L) = P(H | E, L, theta)P(theta) # Bayesian analysis
# We are interested in the likelihood that P(H | L, E)

import tensorflow as tf
import tensorflow_probability as tfp
import os

# Disable tensorflow AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

DETERMINISTIC = True
SEED = 1

if DETERMINISTIC:
    tf.random.set_seed(SEED)

WINDOW = 10
SAMPLES = I = 100
ENTITIES = J = 30
HEURISTICS = K = 10
LOSSES = L = 2

# Model
ALPHA = tf.ones(shape = (K), name="ALPHA")
BETA = tf.ones(shape = (J, K), name="BETA")
GAMMA = tf.ones(shape = (L, K), name="GAMMA")
# print(ALPHA)
# print(BETA)
# print(GAMMA)

# SAMPLE
for i in range(0, SAMPLES):
    # Distribution
    dist_theta = tfp.distributions.Dirichlet(ALPHA)
    dist_phi = tfp.distributions.Dirichlet(BETA)
    dist_psi = tfp.distributions.Dirichlet(GAMMA)

    # Samples
    sample_theta = dist_theta.sample()
    sample_phi = dist_phi.sample()
    sample_psi = dist_psi.sample()

    # Sample biases towards performance class
    sample_psi_bias = sample_psi[0]

    # Calculate initial probabilities for each heuristic (using sample bias)
    # Need log-exp-sum trick here
    probability_h = tf.math.multiply(sample_psi_bias, sample_phi)
    probability_h = tf.math.multiply(probability_h, sample_theta)

    # Selection
    selection = tf.argmax(probability_h, axis=1)
    print(selection)

    # Introduce drift
    if i % WINDOW == 0:
        BIAS = tf.convert_to_tensor([3.0,0,0,0,0,0,0,0,0,0])
        ALPHA = ALPHA + BIAS

