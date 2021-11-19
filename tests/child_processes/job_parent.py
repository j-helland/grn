def job_parent():
    import subprocess
    import torch
    x = torch.randn(32, 512, 512, 3, dtype=torch.float32).cuda()

    proc = subprocess.Popen(['python', 'job_child.py'])
    proc.wait()

if __name__ == '__main__':
    job_parent()