import os
import sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from torch.cuda import device_count
from torch.multiprocessing import spawn

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))
from config.config import Config
if Config.get_config().model.use_latent:
    from modules.train.latent_learner import train, train_distributed
else:
    from modules.train.learner import train, train_distributed

def _get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]

def check_model_config():
    data_config = Config.get_data_config()
    profiler_config = Config.get_profiler_config()
    config = Config.get_config()
    # DATASET
    if data_config.train_cond_dirs is not None:
        print('Conditioning will be loaded from file!')

    if profiler_config.use_profiler is True:
        assert profiler_config.wait is not None
        assert profiler_config.warmup is not None
        assert profiler_config.active is not None
    # GPUs
    print("Cuda GPUs codes set to be used:", config.CUDA_VISIBLE_DEVICES)

def main():
    train_config = Config.get_training_config()
    config = Config.get_config()
    check_model_config()
    replica_count = device_count()

    if replica_count > 1:
        if train_config.batch_size % replica_count != 0:
            raise ValueError(
                f"Batch size {train_config.batch_size} is not evenly divisible by # GPUs {replica_count}."
            )
        train_config.batch_size = train_config.batch_size // replica_count
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, config),
            nprocs=replica_count,
            join=True,
        )

    else:
        train(config)

if __name__ == "__main__": 
    main()
