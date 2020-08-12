#!/usr/bin/env python
import client

import rospy


def init_commu():
    host_ip = rospy.get_param("/commu_node/host_ip")
    port = rospy.get_param('/commu_node/port')
    host = (host_ip, port)
    Client = client.ClientSocket(host)
    return Client


def send_msg(Sock, data):
    Sock.send(data)


def check_received_msg(Sock):
    return Sock.check_recv()
