from __future__ import annotations

import sys as _sys

from bot import app_main as _app_main

# Keep backward compatibility for code/tests importing `main` symbols directly.
_sys.modules[__name__] = _app_main

if __name__ == "__main__":
    _app_main.run()
