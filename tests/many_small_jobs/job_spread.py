import glb

@glb.job(resource_policy='spread')
def job(mem: int) -> None:
    import torch
    x = torch.randn(int(mem * 8e6 // 32), device='cuda:0')

if __name__ == '__main__':
    job(100)
