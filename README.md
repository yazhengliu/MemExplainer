# MemExplainer
This is the Pytorch Implementation of [Towards the Explainability of Temporal Graph Networks via Memory
Backtracking and Topological Attribution](https://openreview.net/forum?id=sKAgLgpujy&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2026%2FConference%2FAuthors%23your-submissions))
## 1. Data

The raw and processed data are stored at [this](https://hkustgz-my.sharepoint.com/:f:/g/personal/yliu533_connect_hkust-gz_edu_cn/IgBLe4o3K1LyTLoyElSmrD3kAXIiYe-dZm3uWVWHvNs7Zqk?e=WbKWwL)

If you want to rebuild human pose-based data from raw datasets, run:

```bash
python data_process/penn_action_process.py
python data_process/process_hmdb51.py
```

## 2. TGNs training
### Link prediction task

To train the temporal graph neural networks, run the following commands. Replace `{data}` with `uci`, `wikipedia`, `reddit`, or `enron`.

```bash
python train_link_prediction.py --config configs/train_link_prediction_{data}.json
```

### Node property prediction task

To train the temporal graph neural networks, run the following commands. Replace `{data}` with `genre`, `reddit`, or `trade`.

```bash
python train_node_prediction.py --config configs/train_node_prediction_{data}.json
```

### Pose-based action classification task
To train the temporal graph neural networks, run the following command. Replace `{data}` with `penn` or `hmdb`.
```bash
python train_video.py --config configs/train_video_{data}.json
```

**TGN Training Parameters**

<table>
  <thead>
    <tr>
      <th width="220">Parameter</th>
      <th>Description</th>
      <th>Common values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td nowrap><code>--embedding_module</code></td>
      <td>Temporal embedding module used by TGN.</td>
      <td><code>graph_sum</code>, <code>graph_attention</code>, <code>identity</code>, <code>time</code></td>
    </tr>
    <tr>
      <td nowrap><code>--aggregator</code></td>
      <td>Aggregates messages for each node.</td>
      <td><code>last</code>, <code>mean</code></td>
    </tr>
    <tr>
      <td nowrap><code>--message_function</code></td>
      <td>Transforms raw messages before memory update.</td>
      <td><code>identity</code>, <code>mlp</code></td>
    </tr>
    <tr>
      <td nowrap><code>--memory_updater</code></td>
      <td>Updates node memory after receiving messages.</td>
      <td><code>gru</code>, <code>rnn</code></td>
    </tr>
    <tr>
      <td nowrap><code>--use_memory</code></td>
      <td>Enables TGN node memory.</td>
      <td><code>true</code>, <code>false</code></td>
    </tr>
    <tr>
      <td nowrap><code>--n_layer</code></td>
      <td>Number of temporal graph embedding layers.</td>
      <td>Default 1</td>
    </tr>
    <tr>
      <td nowrap><code>--n_head</code></td>
      <td>Number of attention heads.</td>
      <td>Default 2</td>
    </tr>
    <tr>
      <td nowrap><code>--n_degree</code></td>
      <td>Number of temporal neighbors sampled per node.</td>
      <td>Default 10</td>
    </tr>
  </tbody>
</table>


