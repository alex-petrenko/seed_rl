import socket

from seed_rl.utils.utils import log


def is_udp_port_available(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', port))
        sock.close()
    except OSError as exc:
        log.warning(f'UDP port {port} cannot be used {str(exc)}')
        return False
    else:
        return True
