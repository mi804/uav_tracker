#!/usr/bin/env python

import base64
import io
import math
import socket
import time


def img2str(file_name):
    f = open(file_name, 'rb')
    f_str = base64.b64encode(f.read())
    f.close()
    return f_str


def str2img(f_str, file_name):
    file_str = open(file_name, 'wb')
    file_str.write(base64.b64decode(f_str))
    file_str.close()


class ClientSocket(object):
    def __init__(self, host_port):
        self.host_port = host_port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('connecting to server')
        self.client_socket.connect(host_port)
        print('connected')

    def send(self, stringData):
        try:
            sizestr = str(len(stringData)).ljust(16)
            self.client_socket.send('^')  # packet start
            self.client_socket.send(sizestr)
            self.client_socket.send(stringData)
            self.client_socket.send('&')  # packet end
        except socket.error:
            try:
                self.client_socket.connect(self.host_port)
            except:
                pass

    def send_img(self, file_name):
        istr = img2str(file_name)
        size = len(istr)
        flag = 'Image:'  # 'image'+count+count*size
        self.send(flag)
        count = int(math.floor(size / 1000) + 1)
        countstr = str(count).ljust(16)
        self.client_socket.send(countstr)
        i = 0
        while 1:
            if (i + 1000) >= len(istr):
                st = istr[i:len(istr)]
                self.send(st)
                break
            st = istr[i:i + 1000]
            self.send(st)
            i += 1000
            print(i)
            time.sleep(0.001)
        print('send success')

    def recv_size(self, sock, count):
        buf = b''
        while count > 0:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def check_recv(self):
        try:
            while 1:
                ss = self.client_socket.recv(1)
                if ss == '^':
                    break

            sizebyte = self.recv_size(self.client_socket, 16)
            if not isinstance(sizebyte, str):
                return -1, ''
            size = int(sizebyte)
            data = self.client_socket.recv(size)
            end_flag = self.client_socket.recv(1)
            if size != len(data) or end_flag != '&':
                return -1, ''
            return size, data
        except (io.BlockingIOError):
            return -1, ''
