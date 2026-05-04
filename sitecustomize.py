import os
import site
import sys


def _ensure_user_site() -> None:
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site) and user_site not in sys.path:
        sys.path.append(user_site)


_ensure_user_site()
