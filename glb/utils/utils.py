import grpc


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