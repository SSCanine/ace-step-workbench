import argparse
import threading
import time
import webbrowser

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(prog="workbench", description="ACE-Step Workbench")
    ap.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to reach it from other devices on your network.")
    ap.add_argument("--port", type=int, default=3470)
    ap.add_argument("--open", action="store_true", help="Open the browser once the server is up.")
    ap.add_argument("--no-ace", action="store_true", help="Do not auto-start the ACE-Step API server.")
    args = ap.parse_args()

    import secrets

    from . import server
    server.AUTOSTART_OVERRIDE = False if args.no_ace else None

    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    if not loopback:
        # Reachable from the network: require a secret token (sent once, then kept as a cookie).
        server.SECURITY["token"] = secrets.token_urlsafe(18)
        url += f"/?token={server.SECURITY['token']}"
        print("\n  WARNING: the Workbench is reachable from your network. Anyone with the link below can use it.")
    print(f"\n  ACE-Step Workbench -> {url}\n")
    if args.open:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
