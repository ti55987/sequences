import numpy as np
import itertools
import matplotlib.pyplot as plt                                                 # visualization
from scipy.stats import chisquare                                               # probability distributions
import pandas as pd                                                             # dataframe management
import os          
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:58:21 2025

This script creates stimulus sequences with controlled delays.

@author: Prof. Dr. Anne Collins, adapted to Python code by Franziska Usée
"""
# function definition
def create_delay_sequence(reps, ns, show_plot=False):
    
    """
    This function creates a stimulus sequence with controlled delays.
    
    Inputs:
        reps: number of repetitions (integer)
        ns  : number of stimuli (integer)
    
    Output:
        seq : generated stimulus sequence (list)
   
    """
    # initialization
    seq       = np.tile(np.arange(1, ns+1), reps)                               # start with simple sequence, repeating the same stimulus order [reps] times
    beta      = 5                                                               # softmax parameter
    criterion = False                                                           # criterion for "good" sequence (as indicated by output from Pearson’s chi-squared test)                                   

    # repeat until criterion is met
    while not criterion:
        
        # initialization of delays and counters
        delays = 1 + np.floor(np.ones(2*ns-1) * reps/2).astype(int)             # max. delay: 2 * set size [ns] -1
        count  = (reps * np.ones(ns)).astype(int) + 1                           # counter: numpy array of shape (ns, ), with all entries being equal to number of repetitions [reps] +1
        seq    = list(range(1, ns+1))                                           # start sequence: [1, 2, ..., ns]
        last   = list(range(1, ns+1))                                           # last presentation index of each stimulus

        # generation of a sequence
        for t in range(ns+1, ns*(reps+1)+1):                                    # start filling positions from (ns+1) to (ns*(reps+1)+1)
            Q = np.zeros(ns)                                                    # initialize "urgency for choosing stimulus" metric for each stimulus; numpy array of shape (ns, )
            L = np.zeros(ns, dtype = int)                                       # initialize last presentation metric for each stimulus, numpy array of shape (ns, )
            
            for i in range(ns):                                                 # iterate through stimuli
                idx_delay = t - last[i]-1                                       # index initialization
                idx_delay = np.clip(idx_delay, 0, len(delays)-1)                # clip index values to maximum of delays 
                Q[i]      = delays[idx_delay] + count[i]                        # value update; compute "urgency" to present stimulus i
                L[i]      = t - last[i]                                         # value update; track how long ago stimulus i was last presented

            # decision rule for which stimulus to present next
            #print(L)
            if np.max(L) == delays.shape[0]:                                    # if maximum in L equals maximum delay (i.e., one stimulus with delay == max. delay),  
                choice = np.argmax(L)                                           # just choose the stimulus with longest delay
            else:
                #print(Q)
                softmax = np.exp(beta * Q)                                      # otherwise, compute softmax probabilities 
                softmax = softmax / np.sum(softmax)
                #print(softmax)
                ps      = np.insert(np.cumsum(softmax), 0, 0)                   # insert 0 in numpy array (first position)
                r       = np.random.rand()                                      # uniform random sampling [0,1]
                choice  = np.where(ps < r)[0][-1]                               # select stimulus for which ps < r   
                #print(choice)
 
            # add selected stimulus to sequence
            seq.append(choice+1)

            # update last, delays, count
            last[choice]              = t                                       # update last occurrence of chosen stimulus
            idx_delay_choice          = L[choice]-1                          
            idx_delay_choice          = np.clip(idx_delay_choice, 0, len(delays)-1)
            delays[idx_delay_choice] -= 1                                       # reduce delays count for stimulus choice
            count[choice]            -= 1                                       # reduce remaining stimulus repetitions

        # analyze the sequence
        alldelays = []
        last_seen = np.zeros(ns, dtype = int)
        dseq      = []

        for t_idx, s in enumerate(seq):
            stim_idx = s-1  
            if last_seen[stim_idx] > 0:
                alldelays.append(t_idx+1 - last_seen[stim_idx])                 # how long since the same stimulus appeared
                dseq.append(s)
            last_seen[stim_idx] = t_idx+1

        alldelays = np.array(alldelays)                                         # sequence with all delays
        dseq      = np.array(dseq)

        # compute the delay distribution
        if len(alldelays) > 0:
            max_delay = np.max(alldelays)                                       # maximum delay
            distr     = np.zeros(max_delay+1, dtype = int)                      # initialization of numpy array of shape (max_delay, ) with all entries being 0

            # count frequency of each delay
            for delay_val in alldelays:
                distr[delay_val] += 1
            
            #print(distr)
            distr = distr[1:]                                                   # remove zero index, delays start from 1
            #print(distr)
            # Pearson's chi-squared test
            # H0: observed delay frequencies are obtained by independent sampling 
            # of N observations from a categorical distribution with given expected 
            # frequencies
            expected = np.mean(distr) * np.ones_like(distr)                     # expected distribution: uniform distribution over all delays 
            _, p     = chisquare(f_obs = distr, 
                                 f_exp = expected)

            # visualization
            if show_plot:
                fig       = plt.figure(figsize = (8,5))  
                plt.clf()                                                           # clear current figure
                plt.plot(distr, "o-")                                               # line plot
                plt.xlabel("Delay")                                                 # x-axis label
                plt.xticks(np.arange(0, len(distr)), np.arange(1, len(distr)+1))
                plt.ylabel("Frequency")
                plt.title("Delay distribution")
                plt.pause(0.1)
                
                # plotting style
                fig.tight_layout()

            # check criterion
            criterion = p > 0.05 and ((np.max(distr) - np.min(distr)) < 2)
            
        else:
            criterion = False

    return np.array(seq)-1

def generate_sequence(num_stims, num_iter_per_stim):
    """
    Generate a sequence where each stimulus appears exactly num_iter_per_stim times
    and consecutive stimuli are different.

    Args:
        num_stims: Number of different stimuli (0 to num_stims-1)
        num_iter_per_stim: Number of times each stimulus should appear

    Returns:
        List of stimuli satisfying the constraints
    """
    # Track remaining count for each stimulus
    remaining_counts = [num_iter_per_stim] * num_stims
    sequence = []

    # Start with a random stimulus
    current_stim = np.random.choice(num_stims)
    sequence.append(current_stim)
    remaining_counts[current_stim] -= 1

    while len(sequence) < num_stims * num_iter_per_stim:
        # Get available stimuli (not the last one and still have remaining count)
        available_stims = [
            stim
            for stim in range(num_stims)
            if stim != sequence[-1] and remaining_counts[stim] > 0
        ]

        if not available_stims:
            # If no valid options, we need to backtrack or use a different strategy
            # This should rarely happen with proper parameters
            raise ValueError("Cannot generate valid sequence with given constraints")

        # Choose next stimulus with preference for those with higher remaining counts
        # This helps balance the sequence and avoid getting stuck
        weights = [remaining_counts[stim] for stim in available_stims]
        next_stim = np.random.choice(
            available_stims, p=np.array(weights) / sum(weights)
        )

        sequence.append(next_stim)
        remaining_counts[next_stim] -= 1

    return sequence

def fix_overunder_repeats(seq, stim_iter_per_block):
    """
    Replaces the last appearances of stims that appear more than stim_iter_per_block+1
    with stims that appear less than stim_iter_per_block+1.
    """
    seq = np.array(seq)
    unique_values, counts = np.unique(seq, return_counts=True)
    over = np.where(counts > (stim_iter_per_block + 1))[0]
    under = np.where(counts < (stim_iter_per_block + 1))[0]

    if len(over) == 0 or len(under) == 0:
        return seq

    # Map from stim to count
    over_stims = unique_values[over]
    under_stims = unique_values[under]
    over_dict = {stim: counts[i] for i, stim in zip(over, over_stims)}
    under_pool = []
    for i, stim in zip(under, under_stims):
        under_pool.extend([stim] * ((stim_iter_per_block + 1) - counts[i]))
    seq_fixed = seq.copy()
    # Traverse from end, replace last over-represented stims
    for idx in reversed(range(len(seq_fixed))):
        stim = seq_fixed[idx]
        if stim in over_stims and over_dict[stim] > (stim_iter_per_block + 1) and under_pool:
            replace_with = under_pool.pop()
            seq_fixed[idx] = replace_with
            over_dict[stim] -= 1
    
    return seq_fixed

def generate_sequence_optimized(num_stims, num_iter_per_stim, max_attempts=100):
    """
    More robust version that can handle edge cases by trying multiple times.
    """
    for attempt in range(max_attempts):
        try:
            return generate_sequence(num_stims, num_iter_per_stim)
        except ValueError:
            continue

    # Fallback: use a more deterministic approach
    return generate_sequence_deterministic(num_stims, num_iter_per_stim)


def generate_sequence_deterministic(num_stims, num_iter_per_stim):
    """
    Deterministic approach that guarantees a valid sequence.
    Creates blocks of stimuli and then shuffles while maintaining constraints.
    """
    # Create base sequence with all stimuli
    base_sequence = []
    for stim in range(num_stims):
        base_sequence.extend([stim] * num_iter_per_stim)

    # Shuffle while maintaining no-consecutive constraint
    sequence = [base_sequence[0]]
    remaining = base_sequence[1:]

    while remaining:
        # Find valid next stimuli
        valid_indices = [i for i, stim in enumerate(remaining) if stim != sequence[-1]]

        if not valid_indices:
            # If stuck, swap with a later element
            for i in range(len(remaining)):
                if remaining[i] != sequence[-1]:
                    # Move this element to a random valid position
                    valid_pos = np.random.choice(
                        [
                            j
                            for j in range(len(remaining))
                            if remaining[j] != remaining[i]
                        ]
                    )
                    remaining[i], remaining[valid_pos] = (
                        remaining[valid_pos],
                        remaining[i],
                    )
                    break
            valid_indices = [
                i for i, stim in enumerate(remaining) if stim != sequence[-1]
            ]

        # Choose random valid option
        chosen_idx = np.random.choice(valid_indices)
        sequence.append(remaining[chosen_idx])
        remaining.pop(chosen_idx)

    return sequence


def shuffle_with_mask(arr, mask):
    """
    Shuffle array where mask[i] = True means keep arr[i] unchanged
    mask[i] = False means this element can be shuffled
    """
    arr = arr.copy()

    # Get indices that can be shuffled
    shuffleable_indices = np.where(~mask)[0]

    if len(shuffleable_indices) <= 1:
        return arr  # Nothing to shuffle

    # Extract values at shuffleable positions
    shuffleable_values = arr[shuffleable_indices]
    np.random.shuffle(shuffleable_values)

    # Put shuffled values back
    arr[shuffleable_indices] = shuffleable_values

    return arr


def swap_by_indices(arr, target_value, target_index):
    """
    Swap elements at two specific indices
    """
    arr = arr.copy()
    curr_index = np.where(arr == target_value)[0][0]
    arr[curr_index], arr[target_index] = arr[target_index], arr[curr_index]
    return arr

def generate_kv_mapping(num_keys, num_values):
    base_array = np.random.permutation(num_values)
    residual = num_keys - num_values
    # Draw 2 additional numbers from range 1 to num_food
    additional_values = np.random.choice(
        range(num_values), size=residual, replace=False
    )

    return np.concatenate([base_array, additional_values])  

def generate_seq_pair(num_stims, num_iter_per_stim, num_directions=4):
    # stim_seq = np.repeat(np.arange(num_stims), num_iter_per_stim)
    # key_dir = np.tile(np.arange(num_directions), len(stim_seq) // 4)
    # for i in [0, 1]:
    #     # best effort to avoid consecutive in stim seq first and then key_dir
    #     paired_data = shuffle_with_consecutive_check(stim_seq, key_dir, i)
    #     stim_seq, key_dir = zip(*paired_data)
    stim_seq = list(
        itertools.chain.from_iterable(
            np.random.permutation(num_stims) for _ in range(num_iter_per_stim)
        )
    )
    key_dir = list(
        itertools.chain.from_iterable(
            np.random.permutation(num_directions) for _ in range(len(stim_seq) // num_directions)
        )
    )

    return np.array(stim_seq), np.array(key_dir)