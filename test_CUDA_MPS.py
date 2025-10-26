import torch
import platform


def fmt_bytes(n):
    # Returns human-friendly GiB
    return f"{n / (1024 ** 3):.2f} GiB"


def main():
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")

    # CUDA check
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        cuda_version = getattr(torch.version, 'cuda', None)
        print(f"  CUDA toolkit: {cuda_version}")
        print(f"  Device count: {torch.cuda.device_count()}")
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        cap = torch.cuda.get_device_capability(idx)
        print(f"  Current device: {idx} | Name: {props.name}")
        print(f"  Compute capability: {cap[0]}.{cap[1]}")
        print(f"  Total memory: {fmt_bytes(props.total_memory)}")

    # Apple Silicon MPS check
    has_mps_backend = hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()
    has_mps_runtime = has_mps_backend and torch.backends.mps.is_available()
    print(f"MPS built: {has_mps_backend} | MPS available (runtime): {has_mps_runtime}")

    # Decide which device would be used by default
    device = (
        torch.device('cuda') if has_cuda else
        torch.device('mps') if has_mps_runtime else
        torch.device('cpu')
    )
    print(f"Selected default device: {device}")

    # Tiny tensor test on the selected device
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"Tensor device: {x.device} | sum: {x.sum().item()}")
    except Exception as e:
        print(f"Tensor test failed on {device}: {e}")


if __name__ == "__main__":
    main()