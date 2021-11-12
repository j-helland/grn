from grpc_tools import protoc

protoc.main((
    '',
    '-I./glb/grpc_resources',
    '--python_out=./glb/grpc_resources',
    '--grpc_python_out=./glb/grpc_resources',
    './glb/grpc_resources/master.proto',
))