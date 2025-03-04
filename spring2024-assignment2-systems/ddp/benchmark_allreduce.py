import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def test_all_reduce(rank, world_size, tensor_size, result_dict, backend, device):
    setup(rank, world_size, backend)
    if device == 'cuda':
        print("Using CUDA")
        data = torch.randn(tensor_size).cuda(rank)  # Create a tensor with random values
    else:
        data = torch.randn(tensor_size)
    # print(f"Rank {rank} data before all-reduce: {data.numpy()}")
    torch.cuda.synchronize()  # Synchronize all GPUs before starting the timer
    start_time = time.time()
    dist.all_reduce(data, async_op=False)  # Perform all-reduce operation
    elapsed_time = time.time() - start_time
    torch.cuda.synchronize()  # Synchronize all GPUs after the operation
    # print(f"Rank {rank} data after all-reduce: {data.numpy()}")
    if rank == 0:  # Only report timing from rank 0
        result_dict[(world_size, tensor_size)] = elapsed_time
    dist.destroy_process_group()  # Cleanup the process group

def plot_results(final_results):
    for (backend, device), results in final_results.items():
        # Prepare data for plotting
        world_sizes = sorted(set(ws for ws, _ in results))
        tensor_sizes = sorted(set(ts for _, ts in results))
        tensor_size_labels = [f"{ts / (1024**2) * 4:.1f} MB" for ts in tensor_sizes]

        # Create a plot for each backend and device
        plt.figure(figsize=(10, 6))
        for world_size in world_sizes:
            times = [results.get((world_size, ts), None) for ts in tensor_sizes]
            plt.plot(
                tensor_size_labels, 
                times, 
                marker='o', 
                label=f"World Size: {world_size}"
            )

        # Add titles and labels
        plt.title(f"All-Reduce Benchmark Results ({backend}, {device})")
        plt.xlabel("Tensor Size")
        plt.ylabel("Time (seconds)")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"benchmark_{backend}_{device}.png")
        plt.show()


if __name__ == "__main__":


    
    targets = [('gloo', 'cpu'), ('gloo', 'cuda'), ('nccl', 'cuda')]


    world_sizes = [2,4,6]  # Number of processes

    data_sizes = [512, 1024, 10240, 51200, 102400, 512000, 1024000]  # in KB

    tensor_sizes = [size * 1024 // 4 for size in data_sizes]
    final_results = {}
    
    for backend, device in targets:
        results = {}
        print(f"\nRunning with backend {backend} and device {device}")
        for world_size in world_sizes:
            for tensor_size in tensor_sizes:
                print(f"Running with world size {world_size} and tensor size {tensor_size} MB")
                
                result_dict = mp.Manager().dict()
                
                mp.spawn(
                    test_all_reduce, 
                    args=(world_size, tensor_size, result_dict, backend, device), 
                    nprocs=world_size, 
                    join=True
                )
                results.update(result_dict)
        final_results[(backend, device)] = results

    for (backend, device), results in final_results.items():
        print(f"\nBackend: {backend}, Device: {device}")
        print("\nBenchmark Results:")
        for (world_size, tensor_size), elapsed_time in results.items():
            tensor_size_mb = tensor_size * 4 / (1024**2)
            print(f"World Size: {world_size}, Tensor Size: {tensor_size_mb:.1f} MB, Time: {elapsed_time:.6f} seconds")    

    print(final_results)
    plot_results(final_results)