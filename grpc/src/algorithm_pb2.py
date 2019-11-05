# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: algorithm.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='algorithm.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x0f\x61lgorithm.proto\"\x1c\n\x0cHelloRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x1d\n\nHelloReply\x12\x0f\n\x07message\x18\x01 \x01(\t\"\x1c\n\x04\x64\x61ta\x12\t\n\x01\x61\x18\x01 \x01(\x05\x12\t\n\x01\x62\x18\x02 \x01(\x05\"\x18\n\x06result\x12\x0e\n\x06result\x18\x01 \x01(\x02\"J\n\x08Knn_data\x12\t\n\x01k\x18\x01 \x01(\x05\x12\x10\n\x08testdata\x18\x02 \x01(\t\x12\x11\n\ttraindata\x18\x03 \x01(\t\x12\x0e\n\x06labels\x18\x04 \x01(\t\" \n\nKnn_result\x12\x12\n\nKnn_result\x18\x01 \x01(\t\":\n\x0cK_means_data\x12\x0b\n\x03num\x18\x01 \x01(\x05\x12\x0c\n\x04kind\x18\x02 \x01(\x05\x12\x0f\n\x07km_data\x18\x03 \x03(\t\"#\n\x0eK_means_result\x12\x11\n\tkm_result\x18\x01 \x01(\t23\n\x07Greeter\x12(\n\x08SayHello\x12\r.HelloRequest\x1a\x0b.HelloReply\"\x00\x32q\n\tCalculate\x12\x17\n\x03\x61\x64\x64\x12\x05.data\x1a\x07.result\"\x00\x12\x19\n\x05multi\x12\x05.data\x1a\x07.result\"\x00\x12\x17\n\x03sub\x12\x05.data\x1a\x07.result\"\x00\x12\x17\n\x03\x64iv\x12\x05.data\x1a\x07.result\"\x00\x32&\n\x03Knn\x12\x1f\n\x03knn\x12\t.Knn_data\x1a\x0b.Knn_result\"\x00\x32\x36\n\x07K_means\x12+\n\x07k_means\x12\r.K_means_data\x1a\x0f.K_means_result\"\x00\x62\x06proto3')
)




_HELLOREQUEST = _descriptor.Descriptor(
  name='HelloRequest',
  full_name='HelloRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='HelloRequest.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=19,
  serialized_end=47,
)


_HELLOREPLY = _descriptor.Descriptor(
  name='HelloReply',
  full_name='HelloReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='HelloReply.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=78,
)


_DATA = _descriptor.Descriptor(
  name='data',
  full_name='data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='a', full_name='data.a', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='data.b', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=80,
  serialized_end=108,
)


_RESULT = _descriptor.Descriptor(
  name='result',
  full_name='result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='result.result', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=134,
)


_KNN_DATA = _descriptor.Descriptor(
  name='Knn_data',
  full_name='Knn_data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='k', full_name='Knn_data.k', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='testdata', full_name='Knn_data.testdata', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='traindata', full_name='Knn_data.traindata', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='labels', full_name='Knn_data.labels', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=210,
)


_KNN_RESULT = _descriptor.Descriptor(
  name='Knn_result',
  full_name='Knn_result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='Knn_result', full_name='Knn_result.Knn_result', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=244,
)


_K_MEANS_DATA = _descriptor.Descriptor(
  name='K_means_data',
  full_name='K_means_data',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num', full_name='K_means_data.num', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='kind', full_name='K_means_data.kind', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='km_data', full_name='K_means_data.km_data', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=246,
  serialized_end=304,
)


_K_MEANS_RESULT = _descriptor.Descriptor(
  name='K_means_result',
  full_name='K_means_result',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='km_result', full_name='K_means_result.km_result', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=306,
  serialized_end=341,
)

DESCRIPTOR.message_types_by_name['HelloRequest'] = _HELLOREQUEST
DESCRIPTOR.message_types_by_name['HelloReply'] = _HELLOREPLY
DESCRIPTOR.message_types_by_name['data'] = _DATA
DESCRIPTOR.message_types_by_name['result'] = _RESULT
DESCRIPTOR.message_types_by_name['Knn_data'] = _KNN_DATA
DESCRIPTOR.message_types_by_name['Knn_result'] = _KNN_RESULT
DESCRIPTOR.message_types_by_name['K_means_data'] = _K_MEANS_DATA
DESCRIPTOR.message_types_by_name['K_means_result'] = _K_MEANS_RESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HelloRequest = _reflection.GeneratedProtocolMessageType('HelloRequest', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREQUEST,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:HelloRequest)
  })
_sym_db.RegisterMessage(HelloRequest)

HelloReply = _reflection.GeneratedProtocolMessageType('HelloReply', (_message.Message,), {
  'DESCRIPTOR' : _HELLOREPLY,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:HelloReply)
  })
_sym_db.RegisterMessage(HelloReply)

data = _reflection.GeneratedProtocolMessageType('data', (_message.Message,), {
  'DESCRIPTOR' : _DATA,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:data)
  })
_sym_db.RegisterMessage(data)

result = _reflection.GeneratedProtocolMessageType('result', (_message.Message,), {
  'DESCRIPTOR' : _RESULT,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:result)
  })
_sym_db.RegisterMessage(result)

Knn_data = _reflection.GeneratedProtocolMessageType('Knn_data', (_message.Message,), {
  'DESCRIPTOR' : _KNN_DATA,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:Knn_data)
  })
_sym_db.RegisterMessage(Knn_data)

Knn_result = _reflection.GeneratedProtocolMessageType('Knn_result', (_message.Message,), {
  'DESCRIPTOR' : _KNN_RESULT,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:Knn_result)
  })
_sym_db.RegisterMessage(Knn_result)

K_means_data = _reflection.GeneratedProtocolMessageType('K_means_data', (_message.Message,), {
  'DESCRIPTOR' : _K_MEANS_DATA,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:K_means_data)
  })
_sym_db.RegisterMessage(K_means_data)

K_means_result = _reflection.GeneratedProtocolMessageType('K_means_result', (_message.Message,), {
  'DESCRIPTOR' : _K_MEANS_RESULT,
  '__module__' : 'algorithm_pb2'
  # @@protoc_insertion_point(class_scope:K_means_result)
  })
_sym_db.RegisterMessage(K_means_result)



_GREETER = _descriptor.ServiceDescriptor(
  name='Greeter',
  full_name='Greeter',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=343,
  serialized_end=394,
  methods=[
  _descriptor.MethodDescriptor(
    name='SayHello',
    full_name='Greeter.SayHello',
    index=0,
    containing_service=None,
    input_type=_HELLOREQUEST,
    output_type=_HELLOREPLY,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GREETER)

DESCRIPTOR.services_by_name['Greeter'] = _GREETER


_CALCULATE = _descriptor.ServiceDescriptor(
  name='Calculate',
  full_name='Calculate',
  file=DESCRIPTOR,
  index=1,
  serialized_options=None,
  serialized_start=396,
  serialized_end=509,
  methods=[
  _descriptor.MethodDescriptor(
    name='add',
    full_name='Calculate.add',
    index=0,
    containing_service=None,
    input_type=_DATA,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='multi',
    full_name='Calculate.multi',
    index=1,
    containing_service=None,
    input_type=_DATA,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='sub',
    full_name='Calculate.sub',
    index=2,
    containing_service=None,
    input_type=_DATA,
    output_type=_RESULT,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='div',
    full_name='Calculate.div',
    index=3,
    containing_service=None,
    input_type=_DATA,
    output_type=_RESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CALCULATE)

DESCRIPTOR.services_by_name['Calculate'] = _CALCULATE


_KNN = _descriptor.ServiceDescriptor(
  name='Knn',
  full_name='Knn',
  file=DESCRIPTOR,
  index=2,
  serialized_options=None,
  serialized_start=511,
  serialized_end=549,
  methods=[
  _descriptor.MethodDescriptor(
    name='knn',
    full_name='Knn.knn',
    index=0,
    containing_service=None,
    input_type=_KNN_DATA,
    output_type=_KNN_RESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_KNN)

DESCRIPTOR.services_by_name['Knn'] = _KNN


_K_MEANS = _descriptor.ServiceDescriptor(
  name='K_means',
  full_name='K_means',
  file=DESCRIPTOR,
  index=3,
  serialized_options=None,
  serialized_start=551,
  serialized_end=605,
  methods=[
  _descriptor.MethodDescriptor(
    name='k_means',
    full_name='K_means.k_means',
    index=0,
    containing_service=None,
    input_type=_K_MEANS_DATA,
    output_type=_K_MEANS_RESULT,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_K_MEANS)

DESCRIPTOR.services_by_name['K_means'] = _K_MEANS

# @@protoc_insertion_point(module_scope)
