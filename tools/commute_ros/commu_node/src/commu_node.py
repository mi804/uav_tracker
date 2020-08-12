#!/usr/bin/env python
# encoding = utf-8
import time

import rospy

from uav_msgs.msg import Tracking

import wifi


class CommuNode:
    def msgs_init(self):  # Server&Publisher Init
        self.tracking_pub = rospy.Publisher("commu_node/Tracking",
                                            Tracking,
                                            queue_size=100)

    def handle_data(self, size, raw_data):
        # print('received:' + raw_data)
        if size >= 0:
            str_list = raw_data.split(':')
            if len(str_list) != 2:
                print('unknown format', str_list)
                return
            type = str_list[0]  # type(str) = Tracking
            str_data = str_list[1]
            print('receive from server: ' + str_data)
            if type == "Tracking":
                temp_list = str_data.split(',')
                tracked = bool(temp_list[0])
                bbox = [int(float(str(x))) for x in temp_list[1:5]]
                width = int(float(str(temp_list[5])))
                height = int(float(str(temp_list[6])))
                # print(tracked, bbox)
                data = Tracking(tracked, bbox, width, height)
                self.tracking_pub.publish(data)
            else:
                print('unknown type!')

    def pub_get_data(self):  # Use pub
        # wifi get
        size, raw_data = wifi.check_received_msg(self.client)
        self.handle_data(size, raw_data)

    def __init__(self):
        # Communication Channel
        self.client = wifi.init_commu()
        self.msgs_init()


def test_callback(req):
    rospy.loginfo('in test service')
    return None


if __name__ == '__main__':
    rospy.init_node('commu_node')
    rospy.loginfo('begin_main')
    commu_node = CommuNode()
    rate = rospy.Rate(0.05)
    while not rospy.is_shutdown():
        commu_node.pub_get_data()
        time.sleep(0.002)
    rospy.spin()
