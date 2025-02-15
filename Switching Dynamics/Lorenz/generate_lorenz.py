import json
import torch
import numpy as np

import matplotlib.pyplot as plt

data_st = json.load(open("lorenz.json"))
D = len(data_st['data'][0][0])
factors_true = torch.FloatTensor(data_st['factors'])
z_true = torch.FloatTensor(data_st['latents'])
data_st = np.array(data_st['data'])

data_st = data_st + data_st[0].std(axis=0) * 0.001 * np.random.randn(10000, 10)  # added
data_st = (data_st - data_st[0].mean(axis=0)) / data_st[0].std(axis=0)  # added
states = np.zeros(z_true.numpy().shape[0:2])
states[z_true.numpy()[:, :, 0] > 0] = 1
states = torch.LongTensor(states)

z_true = z_true[0]
data_st = data_st[0]
states = states[0]

print(data_st.shape, z_true.shape, states.shape)

# Plot the system, color coating each system by state ID
left = np.where(states == 0)[0]
right = np.where(states == 1)[0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(z_true[left, 0], z_true[left, 1], z_true[left, 2], color='red')
ax.scatter(z_true[right, 0], z_true[right, 1], z_true[right, 2], color='blue')
plt.show()

# Reshape into sequences of 20 timesteps
window_len = 7
strides, labels, latents = [], [], []
for i in range(0, data_st.shape[0] - window_len):
    strides.append(data_st[i:i + window_len, :])
    latents.append(z_true[i:i+window_len, :])
    labels.append(states[i:i + window_len])
strides = np.stack(strides)
latents = np.stack(latents)
labels = np.stack(labels)

print(strides.shape, labels.shape, latents.shape)

# Get samples and their context sets
context, query, query_label, z_latents = [], [], [], []
for i in range(window_len + 3, strides.shape[0] - window_len):
    context.append([
        strides[i - window_len - 2],
        strides[i - window_len - 1],
        strides[i - window_len]
    ])

    query.append(strides[i])
    query_label.append(labels[i])
    z_latents.append(latents[i])

context = np.stack(context)
query = np.stack(query)
query_label = np.stack(query_label)
z_latents = np.stack(z_latents)
print(context.shape, query.shape, query_label.shape, z_latents.shape)

np.savez("lorenz_train.npz", domains=context[:8000], queries=query[:8000], labels=query_label[:8000], latents=z_latents[:8000])
np.savez("lorenz_test.npz", domains=context[8000:], queries=query[8000:], labels=query_label[8000:], latents=z_latents[8000:])