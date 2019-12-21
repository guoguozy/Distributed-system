# coding: utf-8

__author__ = 'zhenhang.sun@gmail.com'
__version__ = '1.0.0'

from node import Node

if __name__ == '__main__':

    conf = {'id': 'node_1',
              'addr': ('localhost', 10001),
              'peers': { 'node_2': ('localhost', 10002), 
                         'node_3': ('localhost', 10003)
                       }
            }
     
    node = Node(conf)

    node.run()