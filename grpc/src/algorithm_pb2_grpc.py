# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import algorithm_pb2 as algorithm__pb2


class GreeterStub(object):
  """Service
  The greeting service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.SayHello = channel.unary_unary(
        '/Greeter/SayHello',
        request_serializer=algorithm__pb2.HelloRequest.SerializeToString,
        response_deserializer=algorithm__pb2.HelloReply.FromString,
        )


class GreeterServicer(object):
  """Service
  The greeting service definition.
  """

  def SayHello(self, request, context):
    """Sends a greeting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_GreeterServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'SayHello': grpc.unary_unary_rpc_method_handler(
          servicer.SayHello,
          request_deserializer=algorithm__pb2.HelloRequest.FromString,
          response_serializer=algorithm__pb2.HelloReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Greeter', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class CalculateStub(object):
  """简单算术运算
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.add = channel.unary_unary(
        '/Calculate/add',
        request_serializer=algorithm__pb2.data.SerializeToString,
        response_deserializer=algorithm__pb2.result.FromString,
        )
    self.multi = channel.unary_unary(
        '/Calculate/multi',
        request_serializer=algorithm__pb2.data.SerializeToString,
        response_deserializer=algorithm__pb2.result.FromString,
        )
    self.sub = channel.unary_unary(
        '/Calculate/sub',
        request_serializer=algorithm__pb2.data.SerializeToString,
        response_deserializer=algorithm__pb2.result.FromString,
        )
    self.div = channel.unary_unary(
        '/Calculate/div',
        request_serializer=algorithm__pb2.data.SerializeToString,
        response_deserializer=algorithm__pb2.result.FromString,
        )


class CalculateServicer(object):
  """简单算术运算
  """

  def add(self, request, context):
    """add algorithm
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def multi(self, request, context):
    """multi algorithm
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def sub(self, request, context):
    """substract algorithm
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def div(self, request, context):
    """divide algorithm
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_CalculateServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'add': grpc.unary_unary_rpc_method_handler(
          servicer.add,
          request_deserializer=algorithm__pb2.data.FromString,
          response_serializer=algorithm__pb2.result.SerializeToString,
      ),
      'multi': grpc.unary_unary_rpc_method_handler(
          servicer.multi,
          request_deserializer=algorithm__pb2.data.FromString,
          response_serializer=algorithm__pb2.result.SerializeToString,
      ),
      'sub': grpc.unary_unary_rpc_method_handler(
          servicer.sub,
          request_deserializer=algorithm__pb2.data.FromString,
          response_serializer=algorithm__pb2.result.SerializeToString,
      ),
      'div': grpc.unary_unary_rpc_method_handler(
          servicer.div,
          request_deserializer=algorithm__pb2.data.FromString,
          response_serializer=algorithm__pb2.result.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Calculate', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class KnnStub(object):
  """KNN
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.knn = channel.unary_unary(
        '/Knn/knn',
        request_serializer=algorithm__pb2.Knn_data.SerializeToString,
        response_deserializer=algorithm__pb2.Knn_result.FromString,
        )


class KnnServicer(object):
  """KNN
  """

  def knn(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_KnnServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'knn': grpc.unary_unary_rpc_method_handler(
          servicer.knn,
          request_deserializer=algorithm__pb2.Knn_data.FromString,
          response_serializer=algorithm__pb2.Knn_result.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Knn', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class K_meansStub(object):
  """K-means 
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.k_means = channel.unary_unary(
        '/K_means/k_means',
        request_serializer=algorithm__pb2.K_means_data.SerializeToString,
        response_deserializer=algorithm__pb2.K_means_result.FromString,
        )


class K_meansServicer(object):
  """K-means 
  """

  def k_means(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_K_meansServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'k_means': grpc.unary_unary_rpc_method_handler(
          servicer.k_means,
          request_deserializer=algorithm__pb2.K_means_data.FromString,
          response_serializer=algorithm__pb2.K_means_result.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'K_means', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
