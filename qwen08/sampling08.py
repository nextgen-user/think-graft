"""Stop probability on a fixed prefix, matching llama.cpp's explicit sampler order.
Top-p precedes temperature in this experiment. This is NOT a trajectory replay theorem:
truncation changes the conditional distribution of the non-stop tokens too.
"""
import numpy as np

def stop_probability(stats, delta):
    stats = np.asarray(stats, dtype=np.float32)
    stop = stats[:, 0] + np.asarray(delta, dtype=np.float32)
    other = stats[:, 1:]
    # Rank of the stop among the 20 strongest alternatives; absent from top 20 => zero.
    rank = np.sum(other >= stop[:,None], axis=1)
    candidates = np.concatenate([stop[:,None],other],axis=1)
    sort = np.argsort(-candidates, axis=1,kind='stable')[:,:20]
    values = np.take_along_axis(candidates,sort,axis=1)
    p = np.exp(values-values[:,:1], dtype=np.float32)
    p /= p.sum(axis=1,keepdims=True,dtype=np.float32)
    cum = np.cumsum(p,axis=1,dtype=np.float32)
    # The token that crosses 0.95 is retained; at least one candidate is kept.
    keep = np.concatenate([np.ones((len(p),1),bool),cum[:,:-1] < np.float32(.95)],axis=1)
    scaled = (values-values[:,:1])/np.float32(.6)
    weights = np.exp(scaled,dtype=np.float32)*keep
    is_stop = (sort==0)&keep
    prob = (weights*is_stop).sum(axis=1)/weights.sum(axis=1)
    prob[rank>=20]=0
    return prob.astype(np.float64)
