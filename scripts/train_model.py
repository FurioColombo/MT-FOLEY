import os
import sys
from pathlib import Path
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch.cuda import device_count
from torch.multiprocessing import spawn

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from params.params import params
from train.learner import train, train_distributed

def _get_free_port():
    import socketserver

    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]

def check_model_params():
    # DATASET
    if params['train_cond_dirs'] is not None:
        print('Conditioning will be loaded from file!')

    if params['use_profiler'] is True:
        assert params['wait'] is not None
        assert params['warmup'] is not None
        assert params['active'] is not None
    # GPUs
    print("Cuda GPUs codes set to be used:", params["CUDA_VISIBLE_DEVICES"])

def main():
    check_model_params()
    replica_count = device_count()

    if replica_count > 1:
        if params['batch_size'] % replica_count != 0:
            raise ValueError(
                f"Batch size {params['batch_size']} is not evenly divisible by # GPUs {replica_count}."
            )
        params['batch_size'] = params['batch_size'] // replica_count
        port = _get_free_port()
        spawn(
            train_distributed,
            args=(replica_count, port, params),
            nprocs=replica_count,
            join=True,
        )

    else:
        train(params)

if __name__ == "__main__": 
    main()