"""
Network utilities.
Handles network-related operations like getting IP addresses and building URLs.
"""
import socket
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """
    Get the local IP address of the machine.
    Tries multiple methods to determine the correct IP.
    
    Returns:
        IP address string (defaults to 127.0.0.1 if detection fails)
    """
    # Method 1: Connect to external address (doesn't actually send data)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        if local_ip and not local_ip.startswith('127.'):
            socket.inet_aton(local_ip)  # Validate IP
            logger.info(f"Detected local IP (method 1): {local_ip}")
            return local_ip
    except Exception as e:
        logger.debug(f"Method 1 failed: {str(e)}")
    
    # Method 2: Use hostname
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip and not local_ip.startswith('127.'):
            socket.inet_aton(local_ip)  # Validate IP
            logger.info(f"Detected local IP (method 2): {local_ip}")
            return local_ip
    except Exception as e:
        logger.debug(f"Method 2 failed: {str(e)}")
    
    # Method 3: Parse network interfaces (Linux/Unix)
    try:
        result = subprocess.run(
            ['ip', 'addr', 'show'],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        for line in result.stdout.split('\n'):
            if 'inet ' in line and 'scope global' in line:
                ip = line.strip().split()[1].split('/')[0]
                if not ip.startswith('127.'):
                    socket.inet_aton(ip)  # Validate
                    logger.info(f"Detected local IP (method 3): {ip}")
                    return ip
    except Exception as e:
        logger.debug(f"Method 3 failed: {str(e)}")
    
    # Fallback
    logger.warning("Could not detect local IP, using fallback: 127.0.0.1")
    return "127.0.0.1"


def get_default_backend_url(local_ip: str = None, port: int = 8001) -> str:
    """
    Get the default backend URL using local IP.
    
    Args:
        local_ip: Local IP address (auto-detected if not provided)
        port: Backend port number
        
    Returns:
        Backend URL string
    """
    ip = local_ip or get_local_ip()
    url = f"http://{ip}:{port}/v1"
    logger.debug(f"Backend URL: {url}")
    return url


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def build_url(base_url: str, *paths, **params) -> str:
    """
    Build a URL from base and path components with query parameters.
    
    Args:
        base_url: Base URL
        *paths: Path components to append
        **params: Query parameters
        
    Returns:
        Complete URL string
    """
    from urllib.parse import urljoin, urlencode
    
    # Join paths
    url = base_url
    for path in paths:
        if not url.endswith('/'):
            url += '/'
        url = urljoin(url, path.lstrip('/'))
    
    # Add query parameters
    if params:
        query_string = urlencode(params)
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}{query_string}"
    
    return url


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Check if a port is open on a host.
    
    Args:
        host: Host address
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        True if port is open, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking port {port} on {host}: {str(e)}")
        return False


def get_client_ip(request) -> str:
    """
    Get client IP address from Flask request.
    Handles proxies and load balancers.
    
    Args:
        request: Flask request object
        
    Returns:
        Client IP address
    """
    # Check for forwarded IP (behind proxy/load balancer)
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        # Get first IP in chain
        ip = request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0].strip()
    elif request.environ.get('HTTP_X_REAL_IP'):
        ip = request.environ['HTTP_X_REAL_IP']
    else:
        ip = request.environ.get('REMOTE_ADDR', 'unknown')
    
    return ip
