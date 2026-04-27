import torch
import torchvision.models as models
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()

    # The beauty of ROCm PyTorch: The code is EXACTLY identical to NVIDIA.
    # torch.cuda seamlessly maps to HIP under the hood.
    if not torch.cuda.is_available():
        print("Error: ROCm/CUDA not available in PyTorch!")
        return

    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Setup Model
    print("Initializing ResNet50...")
    model = models.resnet50(weights=None).to(device)
    
    # If multiple GPUs are available, use DataParallel (simple way to test multi-GPU)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs via DataParallel!")
        model = torch.nn.DataParallel(model)
        
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Dummy inputs for benchmarking
    inputs = torch.randn(args.batch_size, 3, 224, 224).to(device)
    targets = torch.randint(0, 1000, (args.batch_size,)).to(device)

    print(f"Warming up for {args.warmup} iterations...")
    for _ in range(args.warmup):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    
    print(f"Running benchmark for {args.iterations} iterations...")
    start_time = time.time()
    
    for _ in range(args.iterations):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    images_per_sec = (args.iterations * args.batch_size) / total_time
    
    print("======================================================")
    print(f" Batch Size: {args.batch_size}")
    print(f" GPUs used:  {num_gpus}")
    print(f" Total Time: {total_time:.3f} s")
    print(f" Throughput: {images_per_sec:.2f} Images/sec")
    print("======================================================")

if __name__ == '__main__':
    main()
