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
      <td>Default value: 1</td>
    </tr>
    <tr>
      <td nowrap><code>--n_head</code></td>
      <td>Number of attention heads.</td>
      <td>Default value: 2</td>
    </tr>
    <tr>
      <td nowrap><code>--n_degree</code></td>
      <td>Number of temporal neighbors sampled per node.</td>
      <td>Default value: 10</td>
    </tr>
  </tbody>
</table>

## 3. TGNs explanation
### Link prediction task

To explain the temporal graph neural networks, run the following commands. Replace `{data}` with `uci`, `wikipedia`, `reddit`, or `enron`.

```bash
python explain_link_prediction.py --config configs/explain_link_prediction_{data}.json
```

### Node property prediction task

To explain the temporal graph neural networks, run the following commands. Replace `{data}` with `genre`, `reddit`, or `trade`.

```bash
python explain_node_prediction.py --config configs/explain_node_prediction_{data}.json
```

### Pose-based action classification task
To explain the temporal graph neural networks, run the following command. Replace `{data}` with `penn` or `hmdb`.
```bash
python explain_video.py --config configs/explain_video_{data}.json
```

<table>
  <thead>
    <tr>
      <th width="280">Parameter</th>
      <th>Description</th>
      <th>Common values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td nowrap><code>--max_depth</code></td>
      <td>Maximum depth of memory backtracking trees.</td>
      <td>task dependent</td>
    </tr>
    <tr>
      <td nowrap><code>--backtrace_child_prune_ratio</code></td>
      <td>Ratio of backtracking children kept at each step. <code>1.0</code> keeps all children.</td>
      <td><code>0.0</code>-<code>1.0</code></td>
    </tr>
    <tr>
      <td nowrap><code>--edge_selection_mode</code></td>
      <td>Strategy for selecting explanatory edges.</td>
      <td><code>ratio</code>, <code>given_number</code></td>
    </tr>
    <tr>
      <td nowrap><code>--select_edge_ratio</code></td>
      <td>Edge selection ratios used when <code>edge_selection_mode=ratio</code>.</td>
      <td>e.g. <code>0.1 0.2 0.3 0.4 0.5</code></td>
    </tr>
    <tr>
      <td nowrap><code>--given_select_numbers</code></td>
      <td>Exact edge counts used when <code>edge_selection_mode=given_number</code>.</td>
      <td>e.g. <code>1 2 3 4 5</code></td>
    </tr>
    <tr>
      <td nowrap><code>--verbose_debug</code></td>
      <td>Prints detailed attribution logs.</td>
      <td><code>true</code>, <code>false</code></td>
    </tr>
  </tbody>
</table>


