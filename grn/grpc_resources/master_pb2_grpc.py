# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import master_pb2 as master__pb2


class GPUMasterStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RequestGPU = channel.unary_unary(
                '/grn.GPUMaster/RequestGPU',
                request_serializer=master__pb2.JobType.SerializeToString,
                response_deserializer=master__pb2.GPU.FromString,
                )
        self.CompleteJob = channel.unary_unary(
                '/grn.GPUMaster/CompleteJob',
                request_serializer=master__pb2.JobProfile.SerializeToString,
                response_deserializer=master__pb2.Empty.FromString,
                )
        self.JobTypeExists = channel.unary_unary(
                '/grn.GPUMaster/JobTypeExists',
                request_serializer=master__pb2.JobType.SerializeToString,
                response_deserializer=master__pb2.BoolResponse.FromString,
                )


class GPUMasterServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RequestGPU(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CompleteJob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def JobTypeExists(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GPUMasterServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RequestGPU': grpc.unary_unary_rpc_method_handler(
                    servicer.RequestGPU,
                    request_deserializer=master__pb2.JobType.FromString,
                    response_serializer=master__pb2.GPU.SerializeToString,
            ),
            'CompleteJob': grpc.unary_unary_rpc_method_handler(
                    servicer.CompleteJob,
                    request_deserializer=master__pb2.JobProfile.FromString,
                    response_serializer=master__pb2.Empty.SerializeToString,
            ),
            'JobTypeExists': grpc.unary_unary_rpc_method_handler(
                    servicer.JobTypeExists,
                    request_deserializer=master__pb2.JobType.FromString,
                    response_serializer=master__pb2.BoolResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'grn.GPUMaster', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GPUMaster(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RequestGPU(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grn.GPUMaster/RequestGPU',
            master__pb2.JobType.SerializeToString,
            master__pb2.GPU.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CompleteJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grn.GPUMaster/CompleteJob',
            master__pb2.JobProfile.SerializeToString,
            master__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def JobTypeExists(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/grn.GPUMaster/JobTypeExists',
            master__pb2.JobType.SerializeToString,
            master__pb2.BoolResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
