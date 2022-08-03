import numpy as np
import numpy.random as rd


def createWeightMask(epsilon, noRows, noCols, hp):
    # generate an Erdos Renyi sparse weights mask
    rng = np.random.RandomState(hp['seed'] + 1000)
    mask_weights = (rng.rand(noRows, noCols) < epsilon).astype(int)
    noParameters = np.sum(mask_weights)
    return [noParameters, mask_weights]


def Pruning_Algorithm(weights, maxepochs, freq, current_itr, prob):
    # Sharan Narang et al.(2017) Exploring sparsity in recurrent neural networks
    # arXiv:1704.05119

    values = np.sort(abs(weights).ravel())
    q = values[int((1 - prob) * values.shape[0])]
    start_itr = 1 * freq
    ramp_itr = round(0.25 * maxepochs) * freq
    end_itr = round(0.5 * maxepochs) * freq
    theta = 2 * q * freq / (2 * (ramp_itr - start_itr) + 3 * (end_itr - ramp_itr))
    phi = 1.5 * theta

    if current_itr < ramp_itr:
        epsilon = theta * (current_itr - start_itr + 1) / freq
    else:
        epsilon = (theta * (ramp_itr - start_itr + 1) \
                  + phi * (current_itr - ramp_itr + 1)) / freq

    pruning_weights = abs(weights.copy())
    pruning_weights[pruning_weights > epsilon] = 1
    pruning_weights[pruning_weights != 1] = 0

    return pruning_weights


def iterative_pruning(weights, s_f, epoch, start_epoch, end_epoch):
    # ZHself._w_min_value,u & Gupta (2017) To prune, or not to prune: exploring the efficacy of
    # pruning for model compression
    # arXiv: 1710.01878
    T = int(end_epoch - start_epoch)
    t = int(epoch - start_epoch)
    s_t = s_f + (1 - s_f) * ((1 - t / T) ** 3)

    values = np.sort(abs(weights).ravel())
    theta = values[int((1 - s_t) * values.shape[0])]

    pruning_weights = abs(weights.copy())
    pruning_weights[pruning_weights > theta] = 1
    pruning_weights[pruning_weights != 1] = 0

    return pruning_weights


# https://github.com/guillaumeBellec/deep_rewiring
def weight_sampler_strict_number(w_0, n_in, n_out, nb_non_zero):
    '''
    Returns a weight matrix and its underlying, variables, and sign matrices needed for rewiring.
    '''

    # Generate the random mask
    is_con_0 = np.zeros((n_in, n_out), dtype=bool)
    ind_in = rd.choice(np.arange(n_in), size=nb_non_zero)
    ind_out = rd.choice(np.arange(n_out), size=nb_non_zero)
    is_con_0[ind_in, ind_out] = True

    # Get the signs
    sign_0 = np.sign(w_0)

    # Define the sparse matrices
    th = np.abs(w_0) * is_con_0
    is_connected = np.greater(th, 0).astype(int)
    w = np.where(is_connected, w_0, np.zeros((n_in, n_out)))

    return w, is_connected, sign_0


# https://github.com/guillaumeBellec/deep_rewiring
def assert_connection_number(theta, targeted_number):
    '''
    Function to check during the tensorflow simulation if the number of
    connection in well defined after each simulation
    '''

    is_con = np.greater(theta, 0)
    nb_is_con = np.sum(is_con.astype(int))
    assert np.equal(nb_is_con, targeted_number), "the number of connection has changed"


# https://github.com/guillaumeBellec/deep_rewiring
def rewiring(theta, weights, target_nb_connection, sign_0, epsilon=1e-12):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:
    '''

    is_con = np.greater(theta, 0).astype(int)
    w = weights * is_con

    n_connected = np.sum(is_con)
    nb_reconnect = target_nb_connection - n_connected
    nb_reconnect = np.max(nb_reconnect, 0)

    reconnect_candidate_coord = np.where(np.logical_not(is_con))

    n_candidates = np.shape(reconnect_candidate_coord)[1]
    reconnect_sample_id = np.random.permutation(n_candidates)[:nb_reconnect]

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        sign = sign_0[s, t]
        w[s, t] = sign * epsilon

    w_mask = np.greater(abs(w), 0).astype(int)

    return w, w_mask, nb_reconnect


# https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


# https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
def rewireMask(weights, noWeights, zeta):
    # rewire weight matrix

    # remove zeta largest negative and smallest positive weights
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)

    largestNegative = values[int((1-zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos +
                                      zeta * (values.shape[0] - lastZeroPos)))]

    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.copy()

    # add zeta random weights
    nrAdd = 0
    noRewires = noWeights - np.sum(rewiredWeights)
    while (nrAdd  < noRewires):
        i = np.random.randint(0, rewiredWeights.shape[0])
        j = np.random.randint(0, rewiredWeights.shape[1])
        if (rewiredWeights[i, j] == 0):
            rewiredWeights[i, j] = 1
            nrAdd += 1

    return [rewiredWeights, weightMaskCore]


def removeMask(weights, noWeights, zeta):
    values = np.sort(abs(weights).ravel())
    lastZeroPos = find_last_pos(values, 0)
    epsilon = values[int(lastZeroPos + zeta * (values.shape[0] - lastZeroPos))]
    rewiredWeights = abs(weights.copy())
    rewiredWeights[rewiredWeights > epsilon] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.copy()

    # add zeta random weights
    nrAdd = 0
    noRewires = noWeights - np.sum(rewiredWeights)
    while (nrAdd < noRewires):
        i = np.random.randint(0, rewiredWeights.shape[0])
        j = np.random.randint(0, rewiredWeights.shape[1])
        if (rewiredWeights[i, j] == 0):
            rewiredWeights[i, j] = 1
            nrAdd += 1

    return [rewiredWeights, weightMaskCore]
