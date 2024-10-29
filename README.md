# Fork of STAMP 

## Files
- `phd_filter.py`: PHD Filter (wrapper) for belief augmentation 
- `phd_env.py`: Variant of `env.py` to use the updated PHD Filter, instead of the Gaussian Process as the persistent monitoring environment


## From STAMP , Gaussian Process (GP)

### Requirements
```bash
python >= 3.9
pytorch >= 1.11
ray >= 2.0
ortools
scikit-image
scikit-learn
scipy
imageio
tensorboard
```

### Training (GP)
1. Set appropriate parameters in `arguments.py -> Arguments`.
2. Run `python driver.py`.

### Evaluation (GP)
1. Set appropriate parameters in `arguments.py -> ArgumentsEval`.
2. Run `python /evals/eval_driver.py`.

## Files
- `arguments.py`: Training and evaluation arguments.
- `driver.py`: Driver of training program, maintain and update the global network.
- `runner.py`: Wrapper of the local network.
- `worker.py`: Interact with environment and collect episode experience.
- `network.py`: Spatio-temporal network architecture.
- `env.py`: Persistent monitoring environment.
- `gaussian_process.py`: Gaussian processes (wrapper) for belief representation.
- `/evals/*`: Evaluation files.
- `/utils/*`: Utility files for graph, target motion, and TSP.
- `/model/*`: Trained model.

