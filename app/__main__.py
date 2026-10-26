"""Start the app with:  python -m app   then open http://127.0.0.1:8010"""

import socket
import sys

import uvicorn

from . import config


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    if port_in_use(config.HOST, config.PORT):
        sys.exit(f"Port {config.PORT} is already used by another program (maybe another project's server).\n"
                 f"Stop it, or start VoxFlow on another port:  PORT=8020 python -m app")
    print(f"VoxFlow running at http://{config.HOST}:{config.PORT}  (Ctrl+C to stop)")
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT)
