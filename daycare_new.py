import logging
from functools import partial
import numpy as np
import elfi



def daycare(t1, t2, t3, n_dcc=29, n_ind=53, n_strains=33, freq_strains_commun=None,
            n_obs=36, time_end=10., batch_size=1, random_state=None):
    r"""
    Parameters
    ----------
    t1 : float or np.array
        Rate of transmission from other individuals at the DCC.
    t2 : float or np.array
        Rate of transmission from the community outside the DCC.
    t3 : float or np.array
        Scaling of co-infection for individuals infected with another strain.
    n_dcc : int, optional
        Number of daycare centers.
    n_ind : int, optional
        Number of individuals in a DCC (same for all).
    n_strains : int, optional
        Number of bacterial strains considered.
    freq_strains_commun : np.array of shape (n_strains,), optional
        Prevalence of each strain in the community outside the DCC. Defaults to 0.1.
    n_obs : int, optional
        Number of individuals sampled from each DCC (same for all).
    time_end : float, optional
        The system is solved using the Direct method until all cases within the batch exceed this.
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    state_obs : np.array
        Observations in shape (batch_size, n_dcc, n_obs, n_strains).

    """
    random_state = random_state or np.random

    t1 = np.asanyarray(t1).reshape((-1, 1, 1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1, 1, 1))
    t3 = np.asanyarray(t3).reshape((-1, 1, 1, 1))

    if freq_strains_commun is None:
        freq_strains_commun = np.full(n_strains, 0.1)

    prob_commun = t2 * freq_strains_commun

    # the state (infection status) is a 4D tensor for computational performance
    state = np.zeros((batch_size, n_dcc, n_ind, n_strains), dtype=np.bool_)

    # time for each DCC in the batch
    time = np.zeros((batch_size, n_dcc))

    n_factor = 1. / (n_ind - 1)
    gamma = 1.  # relative, see paper
    ind_b_dcc = [np.repeat(np.arange(batch_size), n_dcc), np.tile(np.arange(n_dcc), batch_size)]

    while np.any(time < time_end):
        with np.errstate(divide='ignore', invalid='ignore'):
            # probability of sampling a strain; in paper: E_s(I(t))
            prob_strain_adjust = np.nan_to_num(state / np.sum(state, axis=3, keepdims=True))
            prob_strain = np.sum(prob_strain_adjust, axis=2, keepdims=True)

        # Which individuals are already infected:
        intrainfect_rate = t1 * (np.tile(prob_strain, (1, 1, n_ind, 1)) -
                                 prob_strain_adjust) * n_factor + 1e-9

        # init prob to get infected, same for all
        hazards = intrainfect_rate + prob_commun  # shape (batch_size, n_dcc, 1, n_strains)

        # co-infection, depends on the individual's state
        # hazards = np.tile(hazards, (1, 1, n_ind, 1))
        any_infection = np.any(state, axis=3, keepdims=True)
        hazards = np.where(any_infection, t3 * hazards, hazards)

        # (relative) probability to be cured
        hazards[state] = gamma

        # normalize to probabilities
        inv_sum_hazards = 1. / np.sum(hazards, axis=(2, 3), keepdims=True)
        probs = hazards * inv_sum_hazards

        # times until next transition (for each DCC in the batch)
        delta_t = random_state.exponential(inv_sum_hazards[:, :, 0, 0])
        time = time + delta_t

        # choose transition
        probs = probs.reshape((batch_size, n_dcc, -1))
        cumprobs = np.cumsum(probs[:, :, :-1], axis=2)
        x = random_state.uniform(size=(batch_size, n_dcc, 1))
        ind_transit = np.sum(x >= cumprobs, axis=2)

        # update state, need to find the correct indices first
        ind_transit = ind_b_dcc + list(np.unravel_index(ind_transit.ravel(), (n_ind, n_strains)))
        state[tuple(ind_transit)] = np.logical_not(state[tuple(ind_transit)])

    # observation model: simply take the first n_obs individuals
    state_obs = state[:, :, :n_obs, :]

    return state_obs


def daycare_new(t1, t2, t3, n_obs, n_dcc=29, n_ind=53, n_strains=33, freq_strains_commun=None,
            time_end=10., batch_size=1, random_state=None):
    r"""
    Parameters
    ----------
    t1 : float or np.array
        Rate of transmission from other individuals at the DCC.
    t2 : float or np.array
        Rate of transmission from the community outside the DCC.
    t3 : float or np.array
        Scaling of co-infection for individuals infected with another strain.
    n_obs : list
        List of number of individuals sampled from each DCC. Length is n_dcc.
    n_dcc : int, optional
        Number of daycare centers.
    n_ind : int, optional
        Number of individuals in a DCC (same for all).
    n_strains : int, optional
        Number of bacterial strains considered.
    freq_strains_commun : np.array of shape (n_strains,), optional
        Prevalence of each strain in the community outside the DCC. Defaults to 0.1.
    
    time_end : float, optional
        The system is solved using the Direct method until all cases within the batch exceed this.
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    state_obs : list
        list of length n_dcc with observations in shape (batch_size, n_obs, n_strains).

    """
    random_state = random_state or np.random

    t1 = np.asanyarray(t1).reshape((-1, 1, 1, 1))
    t2 = np.asanyarray(t2).reshape((-1, 1, 1, 1))
    t3 = np.asanyarray(t3).reshape((-1, 1, 1, 1))

    if freq_strains_commun is None:
        freq_strains_commun = np.full(n_strains, 0.1)

    prob_commun = t2 * freq_strains_commun

    # the state (infection status) is a 4D tensor for computational performance
    state = np.zeros((batch_size, n_dcc, n_ind, n_strains), dtype=np.bool_)

    # time for each DCC in the batch
    time = np.zeros((batch_size, n_dcc))

    n_factor = 1. / (n_ind - 1)
    gamma = 1.  # relative, see paper
    ind_b_dcc = [np.repeat(np.arange(batch_size), n_dcc), np.tile(np.arange(n_dcc), batch_size)]

    while np.any(time < time_end):
        with np.errstate(divide='ignore', invalid='ignore'):
            # probability of sampling a strain; in paper: E_s(I(t))
            prob_strain_adjust = np.nan_to_num(state / np.sum(state, axis=3, keepdims=True))
            prob_strain = np.sum(prob_strain_adjust, axis=2, keepdims=True)

        # Which individuals are already infected:
        intrainfect_rate = t1 * (np.tile(prob_strain, (1, 1, n_ind, 1)) -
                                 prob_strain_adjust) * n_factor + 1e-9

        # init prob to get infected, same for all
        hazards = intrainfect_rate + prob_commun  # shape (batch_size, n_dcc, 1, n_strains)

        # co-infection, depends on the individual's state
        # hazards = np.tile(hazards, (1, 1, n_ind, 1))
        any_infection = np.any(state, axis=3, keepdims=True)
        hazards = np.where(any_infection, t3 * hazards, hazards)

        # (relative) probability to be cured
        hazards[state] = gamma

        # normalize to probabilities
        inv_sum_hazards = 1. / np.sum(hazards, axis=(2, 3), keepdims=True)
        probs = hazards * inv_sum_hazards

        # times until next transition (for each DCC in the batch)
        delta_t = random_state.exponential(inv_sum_hazards[:, :, 0, 0])
        time = time + delta_t

        # choose transition
        probs = probs.reshape((batch_size, n_dcc, -1))
        cumprobs = np.cumsum(probs[:, :, :-1], axis=2)
        x = random_state.uniform(size=(batch_size, n_dcc, 1))
        ind_transit = np.sum(x >= cumprobs, axis=2)

        # update state, need to find the correct indices first
        ind_transit = ind_b_dcc + list(np.unravel_index(ind_transit.ravel(), (n_ind, n_strains)))
        state[tuple(ind_transit)] = np.logical_not(state[tuple(ind_transit)])

    # observation model: simply take the first n_obs individuals
    #state_obs = state[:, :, :n_obs, :] #Observations in shape (batch_size, n_dcc, n_obs, n_strains)
    state_obs = [state[:, dcc, :n_obs[dcc], :] for dcc in range(n_dcc)]
    return state_obs


def get_model_new(true_params=None, seed_obs=None, **kwargs):
    """
    Parameters
    ----------
    true_params : list, optional
        Parameters with which the observed data is generated.
    seed_obs : int, optional
        Seed for the observed data generation.

    Returns
    -------
    m : elfi.ElfiModel

    """
    logger = logging.getLogger()
    if true_params is None:
        true_params = [3.6, 0.6, 0.1]

    m = elfi.ElfiModel()
    y_obs = daycare_new(*true_params, n_obs=[37, 31, 19, 30, 31, 22, 16, 23, 11, 11, 25, 28, 19, 21,  9,  7, 21,
       23, 23, 31,  6, 30, 26, 21, 22, 16, 14, 15, 23], random_state=np.random.RandomState(seed_obs), **kwargs)
    sim_fn = partial(daycare_new, **kwargs)
    priors = []
    sumstats = []

    priors.append(elfi.Prior('uniform', 0, 11, model=m, name='t1'))
    priors.append(elfi.Prior('uniform', 0, 2, model=m, name='t2'))
    priors.append(elfi.Prior('uniform', 0, 1, model=m, name='t3'))

    elfi.Simulator(sim_fn, *priors, observed=y_obs, name='DCC')

    sumstats.append(elfi.Summary(ss_shannon_new, m['DCC'], name='Shannon'))
    sumstats.append(elfi.Summary(ss_strains_new, m['DCC'], name='n_strains'))
    sumstats.append(elfi.Summary(ss_prevalence_new, m['DCC'], name='prevalence'))
    sumstats.append(elfi.Summary(ss_prevalence_multi_new, m['DCC'], name='multi'))

    elfi.Discrepancy(distance, *sumstats, name='d')
    elfi.Operation(np.log, m['d'], name='logd')

    logger.info("Generated observations with true parameters "
                "t1: %.1f, t2: %.3f, t3: %.1f, ", *true_params)

    return m


def ss_shannon_new(data):
    r"""
    Parameters
    ----------
    data: list of length n_dcc with each element shaped (batch_size, n_obs, n_strains)
    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    batch_size = data[0].shape[0]
    n_dcc = len(data)
    shannon = np.zeros((batch_size, n_dcc)) 

    total_obs = [np.sum(data[dcc], axis=1, keepdims=True) for dcc in range(n_dcc)] # sum of each strain in each batch
    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = [np.nan_to_num(total_obs[dcc] / np.sum(total_obs[dcc], axis=2, keepdims=True)) for dcc in range(n_dcc)]
    
    for dcc in range(n_dcc):
        temp = proportions[dcc]
        temp[temp == 0] = 1
        proportions[dcc] = temp
    
        shannon[:, dcc] = (-np.sum(proportions[dcc] * np.log(proportions[dcc]), axis=2)) # (batch_size, 1)
    
    return shannon


def ss_shannon(data):
    r"""Calculate the Shannon index of diversity of the distribution of observed strains.

    H = -\sum p \log(p)

    https://en.wikipedia.org/wiki/Diversity_index#Shannon_index

    Parameters
    ----------
    data : np.array of shape (batch_size, n_dcc, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    total_obs = np.sum(data, axis=2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        proportions = np.nan_to_num(total_obs / np.sum(total_obs, axis=3, keepdims=True))
    proportions[proportions == 0] = 1
    shannon = (-np.sum(proportions * np.log(proportions), axis=3))[:, :, 0]

    return shannon

def ss_strains(data):
    """Calculate the number of different strains observed.

    Parameters
    ----------
    data : np.array of shape (batch_size, n_dcc, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    strain_active = np.any(data, axis=2)
    n_strain_obs = np.sum(strain_active, axis=2)  # axis 3 is now 2

    return n_strain_obs

def ss_strains_new(data):
    """Calculate the number of different strains observed.

    Parameters
    ----------
    data : list of length n_dcc with arrays (batch_size, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    batch_size = data[0].shape[0]
    n_dcc = len(data)
    n_strain_obs = np.zeros((batch_size, n_dcc))
    strain_active = [np.any(data[dcc], axis=1) for dcc in range(n_dcc)]
    
    for dcc in range(n_dcc):
        n_strain_obs[:, dcc] = np.sum(strain_active[dcc], axis=1)
    
    return n_strain_obs #returns float?


def ss_prevalence(data):
    """Calculate the prevalence of carriage among the observed individuals.

    Parameters
    ----------
    data : np.array of shape (batch_size, n_dcc, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    any_infection = np.any(data, axis=3)
    n_infected = np.sum(any_infection, axis=2)

    return n_infected / data.shape[2]

def ss_prevalence_new(data):
    """Calculate the prevalence of carriage among the observed individuals.

    Parameters
    ----------
    data : list of length n_dcc with arrays (batch_size, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    batch_size = data[0].shape[0]
    n_dcc = len(data)
    prevalence = np.zeros((batch_size, n_dcc))

    any_infection = [np.any(data[dcc], axis=2) for dcc in range(n_dcc)]
    n_infected = [np.sum(any_infection[dcc], axis=1) for dcc in range(n_dcc)]
    for dcc in range(n_dcc):
        prevalence[:, dcc] = n_infected[dcc] / data[dcc].shape[1]

    return prevalence


def ss_prevalence_multi(data):
    """Calculate the prevalence of multiple infections among the observed individuals.

    Parameters
    ----------
    data : np.array of shape (batch_size, n_dcc, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    n_infections = np.sum(data, axis=3)
    n_multi_infections = np.sum(n_infections > 1, axis=2)

    return n_multi_infections / data.shape[2]

def ss_prevalence_multi_new(data):
    """Calculate the prevalence of multiple infections among the observed individuals.

    Parameters
    ----------
    data : list of length n_dcc with arrays (batch_size, n_obs, n_strains)

    Returns
    -------
    np.array of shape (batch_size, n_dcc)

    """
    batch_size = data[0].shape[0]
    n_dcc = len(data)
    prevalence_multi = np.zeros((batch_size, n_dcc))

    n_infections = [np.sum(data[dcc], axis=2) for dcc in range(n_dcc)]
    n_multi_infections = [np.sum(n_infections[dcc] > 1, axis=1) for dcc in range(n_dcc)]
    
    for dcc in range(n_dcc):
        prevalence_multi[:, dcc] = n_multi_infections[dcc] / data[dcc].shape[1]

    return prevalence_multi


def distance(*summaries, observed):
    """Calculate an L1-based distance between the simulated and observed summaries.

    Follows the simplified single-distance approach in:
    Gutmann M U, Corander J (2016). Bayesian Optimization for Likelihood-Free Inference
    of Simulator-Based Statistical Models. JMLR 17(125):1−47, 2016.

    Parameters
    ----------
    *summaries : k np.arrays of shape (m, n)
    observed : list of k np.arrays of shape (1, n)

    Returns
    -------
    np.array of shape (m,)

    """
    summaries = np.stack(summaries)
    observed = np.stack(observed)
    n_ss, _, n_dcc = summaries.shape

    obs_max = np.max(observed, axis=2, keepdims=True) # largest num observed individuals across all DCCs?
    obs_max = np.where(obs_max == 0, 1, obs_max)

    y = observed / obs_max
    x = summaries / obs_max

    # sort to make comparison more robust
    y = np.sort(y, axis=2)
    x = np.sort(x, axis=2)

    # L1 norm divided by the dimension
    dist = np.sum(np.abs(x - y), axis=(0, 2)) / (n_ss * n_dcc)

    return dist