from __future__ import print_function
import logging

import grpc

import algorithm_pb2
import algorithm_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.

    # 连接 rpc 服务器
    channel = grpc.insecure_channel('localhost:50051')
    # 调用 rpc 服务
    stub = algorithm_pb2_grpc.GreeterStub(channel)
    name = str(input("Please input your user name:"))
    response = stub.SayHello(algorithm_pb2.HelloRequest(name=name))
    print("Greeter client received: ", response.message)
    while(True):
        op1 = int(input(
            "Service Guide: 0-Calculate  1-KNN  2-K-means  Any_other_keys-break:\n"))
        if(op1 == 0):
            stub = algorithm_pb2_grpc.CalculateStub(channel)
            while(True):
                op = int(input(
                    "Service Guide: 0-add 1-substract 2-mutiply 3-divide Any_other_keys-break:\n"))
                if(op != 0 and op != 1 and op != 2 and op != 3):
                    break
                num1 = int(input("The frist number: "))
                num2 = int(input("The second number: "))
                if(op == 0):
                    response = stub.add(algorithm_pb2.data(a=num1, b=num2))
                    print("Result: ", response.result)
                elif(op == 1):
                    response = stub.sub(algorithm_pb2.data(a=num1, b=num2))
                    print("Result: ", response.result)
                elif(op == 2):
                    response = stub.multi(algorithm_pb2.data(a=num1, b=num2))
                    print("Result: ", response.result)
                elif(op == 3):
                    response = stub.div(algorithm_pb2.data(a=num1, b=num2))
                    print("Result: ", response.result)
                else:
                    break
        elif(op1 == 1):
            stub = algorithm_pb2_grpc.KnnStub(channel)
            k = int(input("K(int): for example:2\n"))
            testdata = str(input("testdata(str): for example: 0123\n"))
            traindata = str(
                input("traindata(str): for example: 0021_1254_1234\n"))
            labels = str(input("labels(str): for example: 012\n"))

            response = stub.knn(algorithm_pb2.Knn_data(
                k=k, testdata=testdata, traindata=traindata, labels=labels))
            print("Result: ", response.Knn_result)
        elif(op1 == 2):
            stub = algorithm_pb2_grpc.K_meansStub(channel)
            K_means_tmp = algorithm_pb2.K_means_data()
            k = int(input("num of kinds(int): for example:2\n"))
            K_means_tmp.kind = k
            k = int(input("num of data(int): for example:6\n"))
            K_means_tmp.num = k
            for i in range(0, k):
                data = str(
                    input("Listdata(str): for example: 80_90_99\n"))
                K_means_tmp.km_data.append(data)
            response = stub.k_means(K_means_tmp)
            print("Result: ", response.km_result)
            '''
            6
            88_74_96_85
            92_99_95_94
            91_87_99_95
            78_99_97_81
            88_78_98_84
            100_95_100_92
            '''
        else:
            break


if __name__ == '__main__':
    logging.basicConfig()
    run()
