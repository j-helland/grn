import os

import grpc

from grn.core.constants import GRN_SERVER_TMP_INFO_PATH


__all__ = ['find_free_port', 'is_server_available']


def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(('', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def is_server_available(address: str = 'localhost:50051') -> bool:
    channel = grpc.insecure_channel(address, options=(('grpc.enable_http_proxy', 0),))
    try:
        grpc.channel_ready_future(channel).result(timeout=1)
    except grpc.FutureTimeoutError:
        return False
    return True


def find_gpu_master_address() -> str:
    # Find the GPU Master port
    if os.path.isfile(GRN_SERVER_TMP_INFO_PATH):
        with open(GRN_SERVER_TMP_INFO_PATH, 'r') as grn_file:
            gpu_master_addr = f'localhost:{grn_file.read()}'
    # File containing current server port must exist.
    else:
        raise FileNotFoundError(f'{GRN_SERVER_TMP_INFO_PATH}, serve never called.')
    return gpu_master_addr
