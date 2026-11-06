import sys


class DesktopError(RuntimeError):
    pass


# Everything OS-specific sits behind this: a port adds a module with the same three
# functions (create_session, notify, single_instance_lock). See hotkey.Session.
def backend():
    if sys.platform.startswith("linux"):
        from burnish import desktop_linux

        return desktop_linux
    raise DesktopError(f"Desktop integration is not implemented for {sys.platform}")
