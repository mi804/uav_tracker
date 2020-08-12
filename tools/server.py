#!/usr/bin/env python
import socket
import time
import io
import base64


def img2str(file_name):
    f = open(file_name, 'rb')
    f_str = base64.b64encode(f.read())
    f.close()
    return f_str


def str2img(f_str, file_name):
    file_str = open(file_name, 'wb')
    file_str.write(base64.b64decode(f_str))
    file_str.close()


class ServerSocket(object):
    def __init__(self, address):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setblocking(False)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,
                                      1)
        self.server_socket.bind(address)
        self.server_socket.listen(5)
        print('Waiting for client...')
        while 1:
            try:
                conn, addr = self.server_socket.accept()
                self.conn = conn
                self.ipadd = addr[0]
                print('Accept new connection from :', addr)
                break
            except (io.BlockingIOError, socket.error):
                pass

    def send(self, stringData):
        try:
            sizestr = str(len(stringData)).ljust(16)
            self.conn.send('^'.encode())  # packet start
            self.conn.send(sizestr.encode())
            self.conn.send(stringData.encode())
            self.conn.send('&'.encode())  # packet end
        except socket.error:
            print('connection broken')
            try:
                conn, addr = self.server_socket.accept()
                self.conn = conn
            except socket.error:
                pass

    def receive_img(self, sock, file_name):
        time.sleep(0.1)
        count = int(self.recv_size(sock, 16))
        print(count)
        imgstr = ''
        for i in range(0, count):
            print(i)
            while 1:
                size, data = self.check_receive(sock)
                if size >= 0:
                    imgstr += data
                    break
            time.sleep(0.001)
        str2img(imgstr, file_name)

    def receive_image(self, channel, file_name):
        self.receive_img(self.conn[channel], file_name)

    def recv_size(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf

    def check_recv(self):
        return self.check_receive(self.conn)

    def check_receive(self, conn):
        try:
            while 1:
                ss = conn.recv(1)
                if ss == '^':
                    break
            sizebyte = self.recv_size(conn, 16)
            size = int(sizebyte)
            data = conn.recv(size)
            _ = conn.recv(1)
            return size, data
        except (io.BlockingIOError, socket.error):
            return -1, 0
