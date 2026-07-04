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



## 3. TGNs training
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
