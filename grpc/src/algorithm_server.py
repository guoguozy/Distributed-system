from concurrent import futures
import logging
import time
import grpc

from numpy import *
import operator
from sklearn.cluster import KMeans

import algorithm_pb2
import algorithm_pb2_grpc


class Greeter(algorithm_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        print("%s join!" % request.name)
        return algorithm_pb2.HelloReply(message='Welcome to %s!' % request.name)
        


class Calculate(algorithm_pb2_grpc.CalculateServicer):
    # 加
    def add(self, request, context):
        return algorithm_pb2.result(result=request.a+request.b)
    # 减

    def sub(self, request, context):
        return algorithm_pb2.result(result=request.a-request.b)
    # 乘

    def multi(self, request, context):
        return algorithm_pb2.result(result=request.a*request.b)
    # 除

    def div(self, request, context):
        if request.b == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Cannot divide when the denominator is 0!")
            return algorithm_pb2.result()
        return algorithm_pb2.result(result=request.a/request.b)


class Knn(algorithm_pb2_grpc.KnnServicer):
    def knn(self, request, context):

        testdata = array(list(map(int, request.testdata)))
        labels = array(list(request.labels))
        traindata_l = []
        tmp = request.traindata.split('_')
        for i in range(len(tmp)):
            traindata_l.append(list(map(int, tmp[i])))
        traindata = array(traindata_l)
        # 训练数据个数
        # shape取的事训练数据的第一维，即其行数，也就是训练数据的个数
        traindatasize = traindata.shape[0]  # 3

        # 将测试数据转成和历史数据一样的个数  然后和训练数据相减
        # tile()的意思是给一维的测试数据转为与训练数据一样的行和列的格式
        # [[ 0  1  0  2] [-1 -1 -3 -1] [-1 -1 -1 -1]]
        dif = tile(testdata, (traindatasize, 1)) - traindata
        sqdif = dif ** 2  # [[0 1 0 4] [1 1 9 1] [1 1 1 1]]
        # axis=1 ----> 横向相加的意思
        sumsqdif = sqdif.sum(axis=1)  # [ 5 12  4]
        # 此时sumsqdif以成为一维数组
        # [2.23606798 3.46410162         2.        ]
        distance = sumsqdif ** 0.5
        # sortdistance为测试数据各个训练数据的距离按近到远排序之后的结果
        sortdistance = distance.argsort()  # [2 0 1]
        count = {}
        for i in range(0, request.k):
            vote = labels[sortdistance[i]]  # 2 0
            # vote测试数据最近的K个训练数据的类别
            count[vote] = count.get(vote, 0) + 1
        sortcount = sorted(count.items(), key=operator.itemgetter(
            1), reverse=True)  # [(2, 1), (1, 1)]
        # 2
        return algorithm_pb2.Knn_result(Knn_result=str(sortcount[0][0]))


class K_means(algorithm_pb2_grpc.K_meansServicer):
    def k_means(self, request, context):
        num = request.num
        X_tmp = []
        for i in range(0, num):
            tmp = request.km_data[i].split('_')
            X_tmp.append(list(map(eval, tmp)))
        X = array(X_tmp)
        # 对Kmeans确定类别以后的数据集进行聚类
        kmeans = KMeans(n_clusters=request.kind).fit(X)
        # 根据聚类结果，确定所属类别
        pred = kmeans.predict(X)
        return algorithm_pb2.K_means_result(km_result=str(pred))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    algorithm_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    algorithm_pb2_grpc.add_CalculateServicer_to_server(Calculate(), server)
    algorithm_pb2_grpc.add_KnnServicer_to_server(Knn(), server)
    algorithm_pb2_grpc.add_K_meansServicer_to_server(K_means(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(60*60*24)  # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
