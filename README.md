# MemExplainer
We are currently cleaning the codebase and preparing documentation. 

## 1. Dataset Download
### Pose-based action classification task
We train TGNs on pose-based action classification datasets, including Penn Action and HMDB51.

- Penn Action: https://dreamdragon.github.io/PennAction/
- HMDB51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/

Please download the datasets from their official websites and place them under the `data` folder.

## 2. Data Preprocessing

To convert raw Penn Action frames and labels into temporal graph datasets, run:
```bash
python data_process/penn_action_process.py
```

To convert raw HMDB51 data into temporal graph datasets, run:
```bash
python data_process/process_hmdb51.py
```


## 3. TGNs training
### Pose-based action classification task
To train the temporal graph neural networks, run the following command. Replace {data} with either **penn** or **hmdb**.
```bash
python train_video.py --dataset {data}
```

## 4. Explainability Algorithms
The implementation of the explainability module is currently under preparation. We expect to release the source code and detailed instructions by July 5, 2026, after further code cleanup and validation.
