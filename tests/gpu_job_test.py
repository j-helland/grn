import time
import argparse

import grn


@grn.job()
def gpu_job(mem: int, wait: bool = False) -> None:
    import torch
    x = torch.randn(int(mem * 8e6 // 32), device='cuda:0')
    if wait:
        time.sleep(5)


@gpu_job.profile
def _(mem: int, wait: bool = False) -> None:
    import torch
    x = torch.randn(int(mem * 8e6 // 32), device='cuda:0')
    if wait:
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_mem', type=int)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()

    gpu_job(args.gpu_mem, args.wait)