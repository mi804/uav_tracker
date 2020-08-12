#!/usr/bin/env python
import time

import tools.server as server


def init_commu(host_ip, port):
    host = (host_ip, port)
    Server = server.ServerSocket(host)
    return Server


def send_msg(Sock, tracked, bbox, shape):
    ''' tracked,[x1,y1,x2,y2],(width,height)'''
    data = info2msg(tracked, bbox, shape)
    print('send target: ' + data)
    Sock.send(data)


def check_received_msg(Sock):
    return Sock.check_recv()


def info2msg(tracked, bbox, shape):
    msg_type = 'Tracking'
    msg = msg_type + ':' + str(tracked)
    for i in range(0, len(bbox)):
        msg = msg + ',' + str(int(bbox[i]))
    for i in range(0, len(shape)):
        msg = msg + ',' + str(int(shape[i]))
    return msg


if __name__ == "__main__":
    host_ip = '192.168.43.217'
    port = 9999
    sock = init_commu(host_ip, port)
    tracked = True
    bbox = [0, 10, 111, 222]
    while (True):
        time.sleep(0.01)
        data = info2msg(tracked, bbox)
        send_msg(sock, data)
        print('send:' + data)
