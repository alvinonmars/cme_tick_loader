"""Configuration management for CME Tick Loader"""

import os
import platform


def get_default_base_path():
    """
    Auto-detect default base path based on OS and environment

    Priority:
    1. Environment variable CME_DATA_PATH (if set and non-empty)
    2. Platform-specific default

    Returns:
        str: Default base path for CME futures data

    Supported Platforms:
        - macOS (Darwin): Default path configured for development environment
        - Linux/Ubuntu: /mnt/disk1/cme_futures
        - Windows: Not supported (will use Linux default)

    Examples:
        >>> # Use environment variable
        >>> os.environ['CME_DATA_PATH'] = '/custom/path'
        >>> get_default_base_path()
        '/custom/path'

        >>> # Auto-detect on macOS (development environment)
        >>> get_default_base_path()  # On macOS
        '/Users/alvinma/Desktop/work/data/cme_futures'

        >>> # Auto-detect on Linux
        >>> get_default_base_path()  # On Linux
        '/mnt/disk1/cme_futures'

    Note:
        The macOS default path is configured for the specific development
        environment. Other users should set CME_DATA_PATH environment variable.
    """
    # Check environment variable first
    env_path = os.environ.get('CME_DATA_PATH')
    if env_path and env_path.strip():  # Verify non-empty
        return env_path.strip()

    # Platform-specific defaults
    system = platform.system()
    if system == 'Darwin':  # macOS
        # Default for development environment
        # Other users should set CME_DATA_PATH environment variable
        return '/Users/alvinma/Desktop/work/data/cme_futures'
    elif system == 'Linux':
        return '/mnt/disk1/cme_futures'
    else:
        # Windows and other systems not fully supported
        # Fall back to Linux default - users should set CME_DATA_PATH
        return '/mnt/disk1/cme_futures'