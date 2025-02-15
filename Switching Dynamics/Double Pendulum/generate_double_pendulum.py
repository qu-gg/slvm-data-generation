import json
import torch
import numpy as np

import matplotlib.pyplot as plt


# Import the data
f = open("pendulum.json")
data_st = json.load(f)
D = len(data_st['data'][0][0])

factors_true = np.array(data_st['factors'])
states = np.array(data_st['states'])
z_true = np.array(data_st['latents'])
data_st = np.array(data_st['data'])

print(data_st.shape, z_true.shape, states.shape, factors_true.shape)

# Reshape into sequences of 20 timesteps
window_len = 7
strides, labels, latents = [], [], []

for d, z, s in zip(data_st, z_true, states):
    sts, lbls, lts = [], [], []
    for i in range(0, d.shape[0] - window_len):
        sts.append(d[i:i + window_len, :])
        lbls.append(s[i:i+window_len])
        lts.append(z[i:i + window_len, :])
    strides.append(sts)
    labels.append(lbls)
    latents.append(lts)

strides = np.stack(strides)
latents = np.stack(latents)
labels = np.stack(labels)

print(strides.shape, labels.shape, latents.shape)

# Get samples and their context sets
context, query, query_label, z_latents = [], [], [], []
for s, lb, lt in zip(strides, labels, latents):
    c, q, qlbl, lts = [], [], [], []
    for i in range(window_len + 3, strides.shape[0] - window_len):
        c.append([
            s[i - window_len - 2],
            s[i - window_len - 1],
            s[i - window_len]
        ])

        q.append(s[i])
        qlbl.append(lb[i])
        lts.append(lt[i])

    context.append(c)
    query.append(q)
    query_label.append(qlbl)
    z_latents.append(lts)

context = np.stack(context)
query = np.stack(query)
query_label = np.stack(query_label)
z_latents = np.stack(z_latents)
print(context.shape, query.shape, query_label.shape, z_latents.shape)

# Smush down into one larger set
context = np.reshape(context, [-1, context.shape[2], context.shape[3], context.shape[4]])
query = np.reshape(query, [-1, query.shape[2], query.shape[3]])
query_label = np.reshape(query_label, [-1, context.shape[2]])
z_latents = np.reshape(z_latents, [-1, z_latents.shape[2], z_latents.shape[3]])
print(context.shape, query.shape, query_label.shape, z_latents.shape)



np.savez("double_pendulum_train.npz", domains=context[:8000], queries=query[:8000], labels=query_label[:8000], latents=z_latents[:8000])
np.savez("double_pendulum_test.npz", domains=context[8000:], queries=query[8000:], labels=query_label[8000:], latents=z_latents[8000:])