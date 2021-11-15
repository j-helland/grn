from grpc_tools import protoc

protoc.main((
    '',
    '-I./grn/grpc_resources',
    '--python_out=./grn/grpc_resources',
    '--grpc_python_out=./grn/grpc_resources',
    './grn/grpc_resources/master.proto',
))