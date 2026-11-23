"""Project-owned guard for Python subprocesses; upstream sources are unchanged."""
import os

if os.environ.get("MADGRAPH_STEP_OFFLINE") == "1":
    import ipaddress
    import socket

    def _check_host(host):
        if isinstance(host, bytes):
            host = host.decode("ascii")
        if host is None or host in ("", "localhost"):
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise OSError("MadFieldRunner is offline. Remote network access is disabled.")

    _getaddrinfo = socket.getaddrinfo
    _gethostbyname = socket.gethostbyname
    _gethostbyname_ex = socket.gethostbyname_ex
    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex
    _sendto = socket.socket.sendto

    def getaddrinfo(host, *args, **kwargs):
        _check_host(host)
        return _getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        _check_host(host)
        return _gethostbyname(host)

    def gethostbyname_ex(host):
        _check_host(host)
        return _gethostbyname_ex(host)

    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_host(address[0])
        return _connect(self, address)

    def connect_ex(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_host(address[0])
        return _connect_ex(self, address)

    def sendto(self, *args):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_host(args[-1][0])
        return _sendto(self, *args)

    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex
    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
