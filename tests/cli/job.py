def job():
    import torch
    x = torch.randn(32, 512, 512, 3, device='cuda:0')

if __name__ == '__main__':
    job()