"""Receive position estimates from TCP JSON and UDP binary.

TCP frames are STX (0x02), a UTF-8 JSON object, ETX (0x03).
UDP datagrams are a 16-byte big-endian header plus an 8-byte pose,
with a UTF-8 label appended only after a successful code read.

    py -3 tcp_receiver.py
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import struct
import sys
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import ttk
from typing import Callable

STX = 0x02
ETX = 0x03

# UDP binary header, big-endian: magic, version, message type, payload type,
# spare, payload length, sequence. 2+1+1+1+1+2+8 = 16 bytes.
BINARY_MAGIC = 0x4E3D
BINARY_VERSION = 1
BINARY_MESSAGE_TYPE = 0x10
BINARY_HEADER = struct.Struct(">HBBBBHQ")
BINARY_POSE = struct.Struct(">hhHH")

BG = "#f4f5f7"
CARD = "#ffffff"
INK = "#1c2430"
MUTED = "#5e6978"
LINE = "#e4e7ee"
BLUE = "#2f6bff"
GREEN = "#1c8c4a"
AMBER = "#b86e00"
RED = "#c43838"
GRAY = "#8b93a0"

FIELDS = ("message_count", "label", "x", "y", "heading", "confidence", "data_age")


def parse_server_address(text: str, default_port: int = 2112) -> tuple[str, int]:
    raw = text.strip()
    if not raw:
        raise ValueError("Enter a server address, for example 192.168.0.1:2112")

    if raw.startswith("["):
        end = raw.find("]")
        if end < 0:
            raise ValueError("Invalid IPv6 address")
        host = raw[1:end].strip()
        rest = raw[end + 1 :]
        if not rest:
            return host, default_port
        if not rest.startswith(":"):
            raise ValueError("Use [IPv6]:port")
        return host, _parse_port(rest[1:])

    if raw.count(":") == 1:
        host, port_text = raw.rsplit(":", 1)
        host = host.strip()
        if port_text.isdigit():
            if not host:
                raise ValueError("Server address is missing a host")
            return host, _parse_port(port_text)
    elif raw.count(":") > 1:
        raise ValueError("Use host:port, or [IPv6]:port")

    host = raw.strip()
    if not host:
        raise ValueError("Server address is missing a host")
    return host, default_port


def _parse_port(text: str) -> int:
    if not text.isdigit():
        raise ValueError("Port must be a number from 1 to 65535")
    port = int(text)
    if not 1 <= port <= 65535:
        raise ValueError("Port must be a number from 1 to 65535")
    return port


def route_ipv4(peer: str = "192.168.0.1") -> str | None:
    """Local address Windows would use to reach the device."""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((peer, 2112))
            ip = probe.getsockname()[0]
        finally:
            probe.close()
    except OSError:
        return None
    if not ip or ip.startswith("127."):
        return None
    return ip


def local_ipv4_addresses() -> list[str]:
    found: list[str] = []

    def add(ip: str) -> None:
        if ip and not ip.startswith("127.") and ip not in found:
            found.append(ip)

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.168.0.1", 2112))
            add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return found


class FrameParser:
    """Split a TCP byte stream into STX ... ETX payloads."""

    def __init__(self, max_frame: int = 1_048_576) -> None:
        self._buf = bytearray()
        self.max_frame = max_frame

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        frames: list[bytes] = []
        while True:
            start = self._buf.find(bytes((STX,)))
            if start < 0:
                self._buf.clear()
                break
            if start:
                del self._buf[:start]
            end = self._buf.find(bytes((ETX,)), 1)
            if end < 0:
                if len(self._buf) > self.max_frame:
                    del self._buf[0]
                break
            frames.append(bytes(self._buf[1:end]))
            del self._buf[: end + 1]
        return frames


class BinaryError(Exception):
    """A UDP datagram is not a valid binary estimate."""


def decode_binary_datagram(data: bytes) -> tuple[dict, str | None]:
    """Decode one UDP estimate.

    Heading and confidence share one big-endian 16-bit word: the top 12 bits
    are a signed heading in 0.1 degree steps, and the low 4 bits are confidence.
    X and Y are signed 0.1 mm steps. Data age is unsigned 10 µs steps.
    """
    if len(data) < BINARY_HEADER.size + BINARY_POSE.size:
        raise BinaryError(f"Datagram is too short ({len(data)} bytes)")

    magic, version, message_type, payload_type, spare, payload_length, sequence = BINARY_HEADER.unpack_from(data)
    if magic != BINARY_MAGIC:
        raise BinaryError(f"Bad magic 0x{magic:04X}, expected 0x4E3D")
    if message_type != BINARY_MESSAGE_TYPE or payload_type != 0:
        raise BinaryError(
            f"Unsupported message type 0x{message_type:02X} payload type {payload_type}"
        )
    if payload_length < BINARY_POSE.size:
        raise BinaryError(f"Payload is {payload_length} bytes, need at least 8")
    if len(data) != BINARY_HEADER.size + payload_length:
        raise BinaryError(
            f"payload_length {payload_length} does not match {len(data) - BINARY_HEADER.size} payload bytes"
        )

    x_raw, y_raw, heading_word, age_raw = BINARY_POSE.unpack_from(data, BINARY_HEADER.size)
    heading_units = heading_word >> 4
    if heading_units & 0x800:
        heading_units -= 0x1000
    confidence = heading_word & 0x0F
    label_bytes = data[BINARY_HEADER.size + BINARY_POSE.size :]
    if label_bytes:
        try:
            label = label_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryError(f"Label is not UTF-8: {exc}") from exc
    else:
        label = ""

    x_mm = round(x_raw / 10, 1)
    y_mm = round(y_raw / 10, 1)
    heading = round(heading_units / 10, 1)
    data_age = age_raw * 10
    no_read = label == "" and x_mm == 0 and y_mm == 0 and heading == 0 and confidence == 0 and data_age == 0
    warning = None
    if version != BINARY_VERSION:
        warning = f"Protocol version {version}, expected {BINARY_VERSION}"
    if spare != 0:
        spare_note = f"spare is {spare}, expected 0"
        warning = f"{warning}; {spare_note}" if warning else spare_note

    message = {
        "sequence": sequence,
        "message_count": sequence,
        "label": label,
        "x": x_mm,
        "y": y_mm,
        "heading": heading,
        "confidence": confidence,
        "data_age": data_age,
        "no_read": no_read,
        "magic": f"0x{magic:04X}",
        "version": version,
        "message_type": f"0x{message_type:02X}",
        "payload_type": payload_type,
        "payload_length": payload_length,
    }
    return message, warning


def build_binary_datagram(
    sequence: int,
    x_mm: float = 0,
    y_mm: float = 0,
    heading_deg: float = 0,
    confidence: int = 0,
    data_age_us: int = 0,
    label: str | None = None,
    *,
    version: int = BINARY_VERSION,
    message_type: int = BINARY_MESSAGE_TYPE,
    payload_type: int = 0,
    spare: int = 0,
) -> bytes:
    label_bytes = b"" if not label else label.encode("utf-8")
    heading_bits = int(round(heading_deg * 10)) & 0xFFF
    heading_word = (heading_bits << 4) | (confidence & 0x0F)
    payload = BINARY_POSE.pack(
        int(round(x_mm * 10)),
        int(round(y_mm * 10)),
        heading_word,
        int(round(data_age_us / 10)),
    ) + label_bytes
    header = BINARY_HEADER.pack(BINARY_MAGIC, version, message_type, payload_type, spare, len(payload), sequence)
    return header + payload


def format_hex(data: bytes, limit: int = 96) -> str:
    shown = data[:limit]
    text = " ".join(f"{byte:02X}" for byte in shown)
    if len(data) > limit:
        return f"{text} … ({len(data)} bytes)"
    return text


def is_no_read(message: dict) -> bool:
    label = message.get("label")
    label_text = "" if label is None else str(label).strip()
    if label_text not in ("", "0"):
        return False
    for key in ("x", "y", "heading", "confidence", "data_age"):
        if message.get(key, 0) not in (0, 0.0, "0", None):
            return False
    return True


def counter_note(previous: int | None, current: int, name: str = "message_count") -> str | None:
    if previous is None:
        return None
    delta = current - previous
    if delta == 1:
        return None
    if delta == 0:
        return f"{name} stayed at {current}"
    if delta < 0:
        return f"{name} went backwards ({previous} → {current})"
    missed = delta - 1
    return f"missed {missed} message{'s' if missed != 1 else ''} ({previous} → {current})"


def as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value)


def format_number(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        return text if "." in text else f"{text}.0"
    if isinstance(value, int):
        return f"{value}.0"
    return str(value)


def format_age(microseconds: object) -> str:
    number = as_int(microseconds)
    if number is None:
        return "—" if microseconds is None else str(microseconds)
    if number < 1000:
        return f"{number} µs"
    if number < 1_000_000:
        return f"{number} µs  ({number / 1000:.2f} ms)"
    return f"{number} µs  ({number / 1_000_000:.3f} s)"


class ReceiverEngine:
    """Listen for the device, or connect to it, and emit parsed frames."""

    def __init__(self, events: queue.Queue, source: str = "tcp") -> None:
        self.events = events
        self.source = source
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self.running = False

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self, mode: str, host: str, port: int) -> None:
        if self.is_running():
            return
        self._stop.clear()
        with self._lock:
            self._sockets = []
        self.running = True
        self._thread = threading.Thread(
            target=self._run,
            args=(mode, host, port),
            name="tcp-receiver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sockets = list(self._sockets)
        for sock in sockets:
            _close_socket(sock)

    def _track(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.append(sock)

    def _untrack(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets = [item for item in self._sockets if item is not sock]
        _close_socket(sock)

    def _emit(self, kind: str, **payload: object) -> None:
        payload["source"] = self.source
        self.events.put((kind, payload))

    def _log(self, text: str) -> None:
        self._emit("log", text=text)

    def _status(self, state: str, detail: str) -> None:
        self._emit("status", state=state, detail=detail)

    def _run(self, mode: str, host: str, port: int) -> None:
        try:
            if mode == "listen":
                self._listen(host, port)
            else:
                self._connect_loop(host, port)
        finally:
            self.running = False
            if not self._stop.is_set():
                return
            self._status("stopped", "Disconnected")

    def _listen(self, host: str, port: int) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((host, port))
            server.listen(1)
            server.settimeout(0.3)
        except OSError as exc:
            _close_socket(server)
            self._status("error", f"Cannot listen on {host}:{port}")
            self._log(f"Cannot listen on {host}:{port}: {exc}")
            self.running = False
            return

        self._track(server)
        bound = server.getsockname()
        where = f"{bound[0]}:{bound[1]}"
        self._status("listening", f"Listening on {where}")
        self._log(f"Listening on {where}")
        addresses = local_ipv4_addresses()
        if addresses:
            targets = ", ".join(f"{ip}:{bound[1]}" for ip in addresses)
            self._log(f"Point the device server address at {targets}")
        self._log("If nothing connects, allow Python through the Windows firewall.")

        try:
            while not self._stop.is_set():
                try:
                    client, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop.is_set():
                        self._log(f"Accept failed: {exc}")
                    break
                peer = f"{addr[0]}:{addr[1]}"
                self._track(client)
                self._status("connected", f"Connected, device {peer}")
                self._log(f"Device connected from {peer}")
                try:
                    self._read(client)
                finally:
                    self._untrack(client)
                if self._stop.is_set():
                    break
                self._status("listening", f"Listening on {where}")
                self._log("Device disconnected. Waiting for a new connection.")
        finally:
            self._untrack(server)

    def _connect_loop(self, host: str, port: int) -> None:
        target = f"{host}:{port}"
        delay = 0.0
        while not self._stop.is_set():
            if delay and self._stop.wait(delay):
                break
            self._status("connecting", f"Connecting to {target}")
            try:
                sock = socket.create_connection((host, port), timeout=5)
            except OSError as exc:
                if self._stop.is_set():
                    break
                self._status("connecting", f"No connection to {target}")
                self._log(f"Cannot connect to {target}: {exc}")
                self._log("If the device connects out to this PC, switch Mode to Listen.")
                delay = 2.0
                continue

            delay = 1.0
            self._track(sock)
            self._status("connected", f"Connected to {target}")
            self._log(f"Connected to {target}")
            try:
                self._read(sock)
            finally:
                self._untrack(sock)
            if self._stop.is_set():
                break
            self._log(f"Connection to {target} lost. Reconnecting.")

    def _read(self, sock: socket.socket) -> None:
        parser = FrameParser()
        sock.settimeout(0.5)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass

        while not self._stop.is_set():
            try:
                data = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop.is_set():
                    self._log(f"Read error: {exc}")
                break
            if not data:
                self._log("Peer closed the connection")
                break
            for payload in parser.feed(data):
                self._handle_payload(payload)

    def _handle_payload(self, payload: bytes) -> None:
        if not payload:
            self._log("Empty frame")
            return
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            preview = payload[:24].hex(" ")
            self._log(f"Frame is not UTF-8 JSON ({len(payload)} bytes): {preview}")
            return
        try:
            message = json.loads(text)
        except json.JSONDecodeError as exc:
            preview = text if len(text) <= 160 else text[:160] + "…"
            self._log(f"Invalid JSON ({exc.msg}): {preview}")
            return
        if not isinstance(message, dict):
            self._log("JSON payload is not an object")
            return
        self._emit("frame", message=message, text=text)


class UdpEngine:
    """Listen for binary estimate datagrams."""

    def __init__(self, events: queue.Queue) -> None:
        self.events = events
        self.source = "udp"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sockets: list[socket.socket] = []
        self.running = False

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self, host: str, port: int) -> None:
        if self.is_running():
            return
        self._stop.clear()
        with self._lock:
            self._sockets = []
        self.running = True
        self._thread = threading.Thread(
            target=self._run,
            args=(host, port),
            name="udp-receiver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            sockets = list(self._sockets)
        for sock in sockets:
            _close_socket(sock)

    def _track(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets.append(sock)

    def _untrack(self, sock: socket.socket) -> None:
        with self._lock:
            self._sockets = [item for item in self._sockets if item is not sock]
        _close_socket(sock)

    def _emit(self, kind: str, **payload: object) -> None:
        payload["source"] = self.source
        self.events.put((kind, payload))

    def _log(self, text: str) -> None:
        self._emit("log", text=text)

    def _status(self, state: str, detail: str) -> None:
        self._emit("status", state=state, detail=detail)

    def _run(self, host: str, port: int) -> None:
        try:
            self._listen(host, port)
        finally:
            self.running = False
            if self._stop.is_set():
                self._status("stopped", "Stopped")

    def _listen(self, host: str, port: int) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except OSError:
                pass
            if sys.platform == "win32":
                try:
                    # SIO_UDP_CONNRESET wants the command as a signed 32-bit value.
                    # This ignores ICMP port-unreachable resets so the listener stays up.
                    command = struct.unpack("i", struct.pack("I", 0x9800000C))[0]
                    sock.ioctl(command, False)
                except (OSError, OverflowError, TypeError, AttributeError, ValueError):
                    pass
            sock.bind((host, port))
            sock.settimeout(0.3)
        except OSError as exc:
            _close_socket(sock)
            self._status("error", f"Cannot listen on {host}:{port}")
            self._log(f"Cannot listen on {host}:{port}: {exc}")
            return

        self._track(sock)
        bound = sock.getsockname()
        where = f"{bound[0]}:{bound[1]}"
        advice = route_ipv4()
        self._status("listening", f"Listening on {where}")
        self._log(f"Listening on {where} for binary datagrams")
        if advice:
            self._log(
                f"Broadcasts to 255.255.255.255 are not delivered to this PC. "
                f"Set the device receiver address to {advice} and keep port {bound[1]}."
            )
        last_peer = ""
        datagrams = 0
        rejected = 0
        last_status = 0.0
        try:
            while not self._stop.is_set():
                try:
                    data, addr = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if self._stop.is_set():
                        break
                    self._log(f"Read error: {exc}")
                    continue
                datagrams += 1
                peer = f"{addr[0]}:{addr[1]}"
                if peer != last_peer:
                    last_peer = peer
                    self._log(f"Datagram from {peer}")
                try:
                    message, warning = decode_binary_datagram(data)
                except BinaryError as exc:
                    rejected += 1
                    if rejected <= 2 or rejected % 50 == 0:
                        self._log(f"{exc} · {format_hex(data)}")
                    message = None
                    warning = None
                now = time.monotonic()
                if now - last_status >= 0.25 or datagrams == 1:
                    last_status = now
                    if rejected and rejected == datagrams:
                        detail = f"{datagrams} datagrams from {peer}, none decoded"
                    else:
                        detail = f"Receiving from {peer} — {datagrams} datagrams"
                    self._status("connected", detail)
                if message is None:
                    continue
                if warning:
                    self._log(warning)
                self._emit("frame", message=message, text="", raw=data)
        finally:
            self._untrack(sock)


def _close_socket(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _enable_windows_dpi() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes

            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Estimate Receiver")
        self.geometry("860x820")
        self.minsize(800, 740)
        self.configure(bg=BG)

        self.events: queue.Queue = queue.Queue()
        self.engine = ReceiverEngine(self.events)
        self.udp_engine = UdpEngine(self.events)
        self._address_edited = False
        self._last_frame_at: float | None = None
        self._last_bad_confidence: int | None = None
        self._stats = {
            "tcp": {"times": deque(), "last": None, "frames": 0, "gaps": 0, "last_count": None},
            "udp": {"times": deque(), "last": None, "frames": 0, "gaps": 0, "last_count": None},
        }

        self._build_style()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll)

    def _build_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", padding=4)
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure(
            "Conf.Horizontal.TProgressbar",
            troughcolor="#e6e8ee",
            background=BLUE,
            thickness=8,
        )
        style.map("Action.TButton", background=[("active", "#2458d6"), ("!disabled", BLUE)])

    def _build(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=16, pady=14)

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x")
        tk.Label(
            header,
            text="Estimate Receiver",
            bg=BG,
            fg=INK,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="TCP  0x02 · UTF-8 JSON · 0x03          UDP  binary 0x4E3D, big-endian",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 10))

        connection = self._card(root)
        connection.pack(fill="x", pady=(0, 10))

        tk.Label(connection, text="TCP", bg=CARD, fg=INK, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=14, pady=(12, 8)
        )

        tk.Label(connection, text="Mode", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="w", padx=(14, 6)
        )
        self.mode_var = tk.StringVar(value="Connect")
        self.mode_combo = ttk.Combobox(
            connection,
            textvariable=self.mode_var,
            values=("Connect", "Listen"),
            state="readonly",
            width=12,
            font=("Segoe UI", 10),
        )
        self.mode_combo.grid(row=2, column=0, sticky="ew", padx=(14, 8), pady=(2, 8))
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        tk.Label(connection, text="Server address", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=1, column=1, sticky="w"
        )
        self.address_var = tk.StringVar(value="192.168.0.1:2112")
        self.address_entry = tk.Entry(
            connection,
            textvariable=self.address_var,
            font=("Segoe UI", 11),
            bg="#f3f1ee",
            fg=INK,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#e0ddd8",
            highlightcolor=BLUE,
        )
        self.address_entry.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(2, 8), ipady=5)
        self.address_entry.bind("<Key>", self._on_address_key)
        self.address_entry.bind("<Return>", lambda _event: self._toggle())

        tk.Label(connection, text="Format", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=1, column=2, sticky="w", padx=(0, 8)
        )
        tk.Label(connection, text="JSON", bg=CARD, fg=INK, font=("Segoe UI", 11)).grid(
            row=2, column=2, sticky="w", padx=(0, 12), pady=(2, 8)
        )

        self.action_btn = ttk.Button(connection, text="Connect", style="Action.TButton", command=self._toggle)
        self.action_btn.grid(row=2, column=3, sticky="e", padx=(0, 14), pady=(2, 8))

        status_row = tk.Frame(connection, bg=CARD)
        status_row.grid(row=3, column=0, columnspan=4, sticky="ew", padx=14, pady=(0, 4))
        self.dot = tk.Canvas(status_row, width=14, height=14, bg=CARD, highlightthickness=0)
        self.dot.pack(side="left", padx=(0, 8))
        self._dot_id = self.dot.create_oval(2, 2, 12, 12, fill=GRAY, outline=GRAY)
        self.status_var = tk.StringVar(value="Disconnected")
        tk.Label(status_row, textvariable=self.status_var, bg=CARD, fg=INK, font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )

        self.hint_var = tk.StringVar(
            value="Connects to the server and reads the framed JSON estimates it writes."
        )
        tk.Label(
            connection,
            textvariable=self.hint_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=740,
            justify="left",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=14, pady=(0, 12))

        connection.columnconfigure(1, weight=1)

        udp = self._card(root)
        udp.pack(fill="x", pady=(0, 10))
        tk.Label(udp, text="UDP", bg=CARD, fg=INK, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=14, pady=(12, 8)
        )
        tk.Label(udp, text="Listen address", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=1, column=0, sticky="w", padx=(14, 6)
        )
        self.udp_host_var = tk.StringVar(value="0.0.0.0")
        self.udp_host_entry = tk.Entry(
            udp,
            textvariable=self.udp_host_var,
            font=("Segoe UI", 11),
            bg="#f3f1ee",
            fg=INK,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#e0ddd8",
            highlightcolor=BLUE,
            width=18,
        )
        self.udp_host_entry.grid(row=2, column=0, sticky="ew", padx=(14, 8), pady=(2, 8), ipady=5)
        self.udp_host_entry.bind("<Return>", lambda _event: self._toggle_udp())

        tk.Label(udp, text="Port", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(row=1, column=1, sticky="w")
        self.udp_port_var = tk.StringVar(value="5000")
        self.udp_port_entry = tk.Entry(
            udp,
            textvariable=self.udp_port_var,
            font=("Segoe UI", 11),
            bg="#f3f1ee",
            fg=INK,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#e0ddd8",
            highlightcolor=BLUE,
            width=8,
        )
        self.udp_port_entry.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=(2, 8), ipady=5)
        self.udp_port_entry.bind("<Return>", lambda _event: self._toggle_udp())

        tk.Label(udp, text="Format", bg=CARD, fg=MUTED, font=("Segoe UI", 9)).grid(
            row=1, column=2, sticky="w", padx=(8, 8)
        )
        tk.Label(udp, text="Binary", bg=CARD, fg=INK, font=("Segoe UI", 11)).grid(
            row=2, column=2, sticky="w", padx=(8, 12), pady=(2, 8)
        )
        self.udp_btn = ttk.Button(udp, text="Listen", style="Action.TButton", command=self._toggle_udp)
        self.udp_btn.grid(row=2, column=3, sticky="e", padx=(0, 14), pady=(2, 8))

        udp_status = tk.Frame(udp, bg=CARD)
        udp_status.grid(row=3, column=0, columnspan=4, sticky="ew", padx=14, pady=(0, 4))
        self.udp_dot = tk.Canvas(udp_status, width=14, height=14, bg=CARD, highlightthickness=0)
        self.udp_dot.pack(side="left", padx=(0, 8))
        self._udp_dot_id = self.udp_dot.create_oval(2, 2, 12, 12, fill=GRAY, outline=GRAY)
        self.udp_status_var = tk.StringVar(value="Not listening")
        tk.Label(
            udp_status, textvariable=self.udp_status_var, bg=CARD, fg=INK, font=("Segoe UI", 10, "bold")
        ).pack(side="left")
        device_ip = route_ipv4() or "192.168.0.13"
        tk.Label(
            udp,
            text=(
                f"255.255.255.255 does not reach this PC. On the device, set Receiver address to {device_ip} "
                "and leave the port at 5000. Listen address here stays 0.0.0.0."
            ),
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=780,
            justify="left",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=14, pady=(0, 12))
        udp.columnconfigure(0, weight=1)

        estimate = self._card(root)
        estimate.pack(fill="x", pady=(0, 10))
        title_row = tk.Frame(estimate, bg=CARD)
        title_row.pack(fill="x", padx=14, pady=(12, 8))
        tk.Label(title_row, text="Latest estimate", bg=CARD, fg=INK, font=("Segoe UI", 11, "bold")).pack(side="left")
        self.link_pill = tk.Label(
            title_row,
            text="—",
            bg="#eceff3",
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
        )
        self.link_pill.pack(side="right", padx=(8, 0))
        self.pill = tk.Label(
            title_row,
            text="Waiting",
            bg="#eceff3",
            fg=MUTED,
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=2,
        )
        self.pill.pack(side="right")

        grid = tk.Frame(estimate, bg=CARD)
        grid.pack(fill="x", padx=10, pady=(0, 8))
        self.value_vars: dict[str, tk.StringVar] = {}
        cells = [
            ("message_count", "Message count", "Telegram counter"),
            ("label", "Label", "Detected code"),
            ("confidence", "Confidence", "0 to 15"),
            ("data_age", "Data age", "Age before send"),
            ("x", "X", "Position"),
            ("y", "Y", "Position"),
            ("heading", "Heading", "Degrees"),
        ]
        for index, (key, title, caption) in enumerate(cells):
            row, col = divmod(index, 4)
            cell = tk.Frame(grid, bg=CARD)
            cell.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            title_lbl = tk.Label(cell, text=title, bg=CARD, fg=MUTED, font=("Segoe UI", 8))
            title_lbl.pack(anchor="w")
            var = tk.StringVar(value="—")
            self.value_vars[key] = var
            tk.Label(cell, textvariable=var, bg=CARD, fg=INK, font=("Segoe UI", 16, "bold")).pack(anchor="w")
            caption_lbl = tk.Label(cell, text=caption, bg=CARD, fg=GRAY, font=("Segoe UI", 8))
            caption_lbl.pack(anchor="w")
            if key == "message_count":
                self.count_title = title_lbl
                self.count_caption = caption_lbl
            if key == "confidence":
                self.confidence_bar = ttk.Progressbar(
                    cell,
                    style="Conf.Horizontal.TProgressbar",
                    maximum=15,
                    length=140,
                    mode="determinate",
                )
                self.confidence_bar.pack(anchor="w", pady=(4, 0))
            grid.columnconfigure(col, weight=1)

        self.note_var = tk.StringVar(
            value="Data fields are zero when there is no successful read. The message counter still increments."
        )
        tk.Label(
            estimate,
            textvariable=self.note_var,
            bg=CARD,
            fg=MUTED,
            font=("Segoe UI", 8),
            wraplength=740,
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        stats = tk.Frame(root, bg=BG)
        stats.pack(fill="x", pady=(0, 8))
        self.tcp_stats_var = tk.StringVar(value="TCP  0 frames · — · 0 gaps")
        self.udp_stats_var = tk.StringVar(value="UDP  0 frames · — · 0 gaps")
        self.seen_var = tk.StringVar(value="Last frame  —")
        for var in (self.tcp_stats_var, self.udp_stats_var, self.seen_var):
            tk.Label(stats, textvariable=var, bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=(0, 16))

        lower = tk.Frame(root, bg=BG)
        lower.pack(fill="both", expand=True)
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        json_card = self._card(lower)
        json_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        json_head = tk.Frame(json_card, bg=CARD)
        json_head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(json_head, text="Last message", bg=CARD, fg=INK, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(json_head, text="Copy", command=self._copy_json).pack(side="right")
        self.raw = tk.Text(
            json_card,
            height=7,
            bg="#f7f6f4",
            fg=INK,
            font=("Consolas", 10),
            relief="flat",
            wrap="none",
            padx=8,
            pady=6,
        )
        self.raw.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.raw.insert("1.0", "No message yet.")
        self.raw.configure(state="disabled")
        self._last_json = ""

        log_card = self._card(lower)
        log_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        log_head = tk.Frame(log_card, bg=CARD)
        log_head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(log_head, text="Connection log", bg=CARD, fg=INK, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(log_head, text="Clear", command=self._clear_log).pack(side="right")
        self.log = tk.Text(
            log_card,
            height=7,
            bg=CARD,
            fg=INK,
            font=("Consolas", 9),
            relief="flat",
            wrap="word",
            padx=8,
            pady=4,
        )
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log.configure(state="disabled")
        self._append_log("Idle. Connect TCP, listen for UDP, or run both.")

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE)

    def _on_address_key(self, _event: object) -> None:
        self._address_edited = True

    def _on_mode_changed(self, _event: object | None = None) -> None:
        if self.mode_var.get() == "Listen":
            if not self._address_edited:
                self.address_var.set("0.0.0.0:2112")
            addresses = local_ipv4_addresses()
            shown = ", ".join(addresses) if addresses else "this PC"
            self.hint_var.set(
                f"Waits for the device to connect. Set the device server address to {shown} and this port."
            )
        else:
            if not self._address_edited:
                self.address_var.set("192.168.0.1:2112")
            self.hint_var.set("Connects to the server and reads the framed JSON estimates it writes.")
        self._sync_button()

    def _toggle(self) -> None:
        if self.engine.is_running():
            self.engine.stop()
            self._append_log("TCP  Disconnecting")
            return
        try:
            host, port = parse_server_address(self.address_var.get())
        except ValueError as exc:
            self._set_status("error", str(exc))
            self._append_log(f"TCP  {exc}")
            return
        mode = "listen" if self.mode_var.get() == "Listen" else "connect"
        self._reset_link("tcp")
        self.engine.start(mode, host, port)

    def _toggle_udp(self) -> None:
        if self.udp_engine.is_running():
            self.udp_engine.stop()
            self._append_log("UDP  Stopping")
            return
        host = self.udp_host_var.get().strip()
        if not host:
            self._paint_status(self.udp_dot, self._udp_dot_id, self.udp_status_var, "error", "Enter a listen address")
            self._append_log("UDP  Enter a listen address, for example 0.0.0.0")
            return
        try:
            port = _parse_port(self.udp_port_var.get().strip())
        except ValueError as exc:
            self._paint_status(self.udp_dot, self._udp_dot_id, self.udp_status_var, "error", str(exc))
            self._append_log(f"UDP  {exc}")
            return
        self._reset_link("udp")
        self.udp_engine.start(host, port)

    def _reset_link(self, source: str) -> None:
        stats = self._stats[source]
        stats["times"].clear()
        stats["last"] = None
        stats["frames"] = 0
        stats["gaps"] = 0
        stats["last_count"] = None

    def _on_close(self) -> None:
        self.engine.stop()
        self.udp_engine.stop()
        self.destroy()

    def _poll(self) -> None:
        latest: tuple[dict, str, bytes | None] | None = None
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            source = str(payload.get("source", "tcp"))
            if kind == "frame":
                message = payload["message"]
                assert isinstance(message, dict)
                raw = payload.get("raw")
                self._count_frame(message, source)
                latest = (message, source, raw if isinstance(raw, bytes) else None)
            elif kind == "log":
                prefix = "UDP" if source == "udp" else "TCP"
                self._append_log(f"{prefix}  {payload['text']}")
            elif kind == "status":
                if source == "udp":
                    self._paint_status(
                        self.udp_dot, self._udp_dot_id, self.udp_status_var, str(payload["state"]), str(payload["detail"])
                    )
                else:
                    self._set_status(str(payload["state"]), str(payload["detail"]))
        if latest:
            self._show_frame(latest[0], latest[1], latest[2])
        self._refresh_timing()
        self._sync_button()
        self._sync_udp_button()
        self.after(100, self._poll)

    def _count_frame(self, message: dict, source: str = "tcp") -> None:
        now = time.monotonic()
        stats = self._stats[source]
        stats["frames"] += 1
        stats["times"].append(now)
        stats["last"] = now
        self._last_frame_at = now
        count_key = "sequence" if source == "udp" else "message_count"
        current = as_int(message.get(count_key))
        if current is not None:
            note = counter_note(stats["last_count"], current, count_key)
            stats["last_count"] = current
            if note:
                stats["gaps"] += 1
                prefix = "UDP" if source == "udp" else "TCP"
                self._append_log(f"{prefix}  {note}")
        confidence = as_int(message.get("confidence"))
        if confidence is not None and not 0 <= confidence <= 15 and confidence != self._last_bad_confidence:
            self._last_bad_confidence = confidence
            prefix = "UDP" if source == "udp" else "TCP"
            self._append_log(f"{prefix}  confidence {confidence} is outside 0..15")
        required = ("label", "x", "y", "heading", "confidence", "data_age", count_key)
        missing = [key for key in required if key not in message]
        if missing:
            prefix = "UDP" if source == "udp" else "TCP"
            self._append_log(f"{prefix}  Frame missing " + ", ".join(missing))

    def _show_frame(self, message: dict, source: str = "tcp", raw: bytes | None = None) -> None:
        count_key = "sequence" if source == "udp" else "message_count"
        count = message.get(count_key, message.get("message_count"))
        count_int = as_int(count)
        self.value_vars["message_count"].set(f"{count_int:,}" if count_int is not None else _show(count))
        if source == "udp":
            self.count_title.configure(text="Sequence")
            self.count_caption.configure(text="UDP datagram counter")
            self.link_pill.configure(text="UDP", bg="#efe8fb", fg="#6941c6")
        else:
            self.count_title.configure(text="Message count")
            self.count_caption.configure(text="Telegram counter")
            self.link_pill.configure(text="TCP", bg="#e7eefe", fg=BLUE)
        label = message.get("label")
        self.value_vars["label"].set("—" if label is None or str(label) == "" else str(label))
        self.value_vars["x"].set(_with_unit(message.get("x"), "mm"))
        self.value_vars["y"].set(_with_unit(message.get("y"), "mm"))
        self.value_vars["heading"].set(_with_unit(message.get("heading"), "°"))

        confidence = as_int(message.get("confidence"))
        if confidence is None:
            self.value_vars["confidence"].set(_show(message.get("confidence")))
            self.confidence_bar["value"] = 0
        else:
            self.value_vars["confidence"].set(f"{confidence} / 15")
            self.confidence_bar["value"] = max(0, min(15, confidence))
        self.value_vars["data_age"].set(format_age(message.get("data_age")))

        flagged = message.get("no_read")
        no_read = is_no_read(message) if flagged is None else bool(flagged)
        if no_read:
            self.pill.configure(text="No read", bg="#f8efd8", fg=AMBER)
            self.note_var.set("No successful read. Data fields are zero; the counter still incremented.")
        elif source == "udp":
            self.pill.configure(text="Detected", bg="#e5f6ec", fg=GREEN)
            self.note_var.set(
                "UDP units scaled to mm, degrees, and microseconds. Heading uses the top 12 bits; confidence is the low 4 bits."
            )
        else:
            self.pill.configure(text="Detected", bg="#e5f6ec", fg=GREEN)
            self.note_var.set("Position is in millimetres. Heading is in degrees. Data age is in microseconds.")

        shown = {"source": "UDP" if source == "udp" else "TCP", **message}
        self._last_json = json.dumps(shown, indent=2, ensure_ascii=False)
        if raw:
            self._last_json += "\n\n" + format_hex(raw)
        self.raw.configure(state="normal")
        self.raw.delete("1.0", "end")
        self.raw.insert("1.0", self._last_json)
        self.raw.configure(state="disabled")

    def _refresh_timing(self) -> None:
        now = time.monotonic()
        self.tcp_stats_var.set(self._link_stats("tcp", now))
        self.udp_stats_var.set(self._link_stats("udp", now))
        if self._last_frame_at is None:
            self.seen_var.set("Last frame  —")
            return
        age = now - self._last_frame_at
        if age < 1:
            seen = f"{age * 1000:.0f} ms ago"
        elif age < 60:
            seen = f"{age:.1f} s ago"
        else:
            seen = f"{age / 60:.0f} min ago"
        self.seen_var.set(f"Last frame  {seen}")

    def _link_stats(self, source: str, now: float) -> str:
        stats = self._stats[source]
        cutoff = now - 1.0
        while stats["times"] and stats["times"][0] < cutoff:
            stats["times"].popleft()
        rate = "—" if stats["last"] is None else f"{len(stats['times'])} Hz"
        name = "UDP" if source == "udp" else "TCP"
        return f"{name}  {stats['frames']} frames · {rate} · {stats['gaps']} gaps"

    def _sync_button(self) -> None:
        running = self.engine.is_running()
        listen = self.mode_var.get() == "Listen"
        text = ("Stop" if running else "Listen") if listen else ("Disconnect" if running else "Connect")
        self.action_btn.configure(text=text)
        entry_state = "disabled" if running else "normal"
        self.address_entry.configure(state=entry_state)
        self.mode_combo.configure(state="disabled" if running else "readonly")

    def _sync_udp_button(self) -> None:
        running = self.udp_engine.is_running()
        self.udp_btn.configure(text="Stop" if running else "Listen")
        entry_state = "disabled" if running else "normal"
        self.udp_host_entry.configure(state=entry_state)
        self.udp_port_entry.configure(state=entry_state)

    def _set_status(self, state: str, detail: str) -> None:
        self._paint_status(self.dot, self._dot_id, self.status_var, state, detail)

    def _paint_status(self, dot: tk.Canvas, dot_id: int, var: tk.StringVar, state: str, detail: str) -> None:
        colors = {
            "stopped": GRAY,
            "listening": AMBER,
            "connecting": AMBER,
            "connected": GREEN,
            "error": RED,
        }
        color = colors.get(state, GRAY)
        dot.itemconfigure(dot_id, fill=color, outline=color)
        var.set(detail or "Disconnected")

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"{stamp}  {text}\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 200:
            self.log.delete("1.0", f"{lines - 200}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _copy_json(self) -> None:
        if not self._last_json:
            return
        self.clipboard_clear()
        self.clipboard_append(self._last_json)


def _show(value: object) -> str:
    if value is None:
        return "—"
    return str(value)


def _with_unit(value: object, unit: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{format_number(value)} {unit}"
    return f"{value} {unit}"


def _frame(message: dict) -> bytes:
    return bytes((STX,)) + json.dumps(message).encode("utf-8") + bytes((ETX,))


def self_test() -> int:
    checks: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    parser = FrameParser()
    sample = {
        "message_count": 1024,
        "label": "42",
        "x": 15.2,
        "y": -8.4,
        "heading": 87.5,
        "confidence": 14,
        "data_age": 1340,
    }
    raw = _frame(sample)
    check(parser.feed(raw[:8]) == [], "partial frame waits for ETX")
    frames = parser.feed(raw[8:])
    check(frames == [json.dumps(sample).encode("utf-8")], "split frame reassembles")

    zero = {
        "message_count": 1025,
        "label": "0",
        "x": 0,
        "y": 0,
        "heading": 0,
        "confidence": 0,
        "data_age": 0,
    }
    other = dict(sample, message_count=1026, label="7")
    blob = b"\x00\x01" + _frame(zero) + _frame(other)
    parsed = parser.feed(blob)
    check(len(parsed) == 2, "leading garbage and two frames")
    check(json.loads(parsed[0])["message_count"] == 1025, "first payload")
    check(is_no_read(zero) and not is_no_read(sample), "no-read detection")
    check(counter_note(1024, 1026) == "missed 1 message (1024 → 1026)", "counter gap")
    check(parse_server_address("192.168.0.1:2112") == ("192.168.0.1", 2112), "address parse")
    check(parse_server_address("0.0.0.0:2112") == ("0.0.0.0", 2112), "listen address parse")

    events: queue.Queue = queue.Queue()
    engine = ReceiverEngine(events)
    engine.start("listen", "127.0.0.1", 0)
    port = _wait_until(events, lambda kind, payload: kind == "status" and payload["state"] == "listening")
    detail = str(port["detail"])
    bound_port = int(detail.rsplit(":", 1)[1])

    client = socket.create_connection(("127.0.0.1", bound_port), timeout=2)
    try:
        _wait_until(events, lambda kind, payload: kind == "status" and payload["state"] == "connected")
        client.sendall(_frame(sample)[:10])
        time.sleep(0.15)
        client.sendall(_frame(sample)[10:])
        client.sendall(_frame(zero) + bytes((STX,)) + b"not-json" + bytes((ETX,)))
        received = _collect_frames(events, 2, timeout=3)
    finally:
        client.close()

    check(len(received) == 2, "socket delivered both JSON frames")
    check(received[0]["label"] == "42" and received[0]["x"] == 15.2, "example fields")
    check(received[1]["message_count"] == 1025, "zeroed frame")
    invalid = _wait_until(events, lambda kind, payload: kind == "log" and "Invalid JSON" in str(payload["text"]))
    check("Invalid JSON" in str(invalid["text"]), "invalid frame is logged")

    engine.stop()
    _wait_thread(engine)

    sender_ready = threading.Event()
    holder: dict[str, socket.socket] = {}

    def serve_once() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(3)
        holder["server"] = server
        sender_ready.set()
        try:
            peer, _addr = server.accept()
        except OSError:
            return
        with peer:
            peer.sendall(_frame(sample))
            time.sleep(0.3)

    thread = threading.Thread(target=serve_once, daemon=True)
    thread.start()
    sender_ready.wait(2)
    connect_port = holder["server"].getsockname()[1]
    events2: queue.Queue = queue.Queue()
    client_engine = ReceiverEngine(events2)
    client_engine.start("connect", "127.0.0.1", connect_port)
    try:
        got = _collect_frames(events2, 1, timeout=3)
        check(got[0]["heading"] == 87.5, "connect mode receives a frame")
    finally:
        client_engine.stop()
        _close_socket(holder.get("server"))
        thread.join(timeout=2)

    _enable_windows_dpi()
    app = App()
    try:
        packet = build_binary_datagram(1024, 15.2, -8.4, 87.5, 14, 1340, "42")
        check(packet[:2] == b"\x4e\x3d", "magic is big-endian 0x4E3D")
        check(struct.calcsize(">HBBBBHQ") == 16, "binary header is 16 bytes")
        decoded, warning = decode_binary_datagram(packet)
        check(warning is None, "example datagram has no warning")
        check(decoded["sequence"] == 1024 and decoded["label"] == "42", "binary sequence and label")
        check(decoded["x"] == 15.2 and decoded["y"] == -8.4, "binary position in mm")
        check(decoded["heading"] == 87.5 and decoded["confidence"] == 14, "heading and confidence split")
        check(decoded["data_age"] == 1340 and decoded["no_read"] is False, "binary data age")
        check(packet[16:20] == struct.pack(">hh", 152, -84), "x and y are 0.1 mm big-endian")

        tricky = build_binary_datagram(3, heading_deg=-0.1, confidence=15)
        tricky_msg, _tricky_warn = decode_binary_datagram(tricky)
        check(tricky_msg["heading"] == -0.1 and tricky_msg["confidence"] == 15, "signed heading keeps confidence")

        silent = build_binary_datagram(8)
        silent_msg, _silent_warn = decode_binary_datagram(silent)
        check(silent_msg["no_read"] is True and silent_msg["label"] == "", "zeroed datagram omits the label")
        check(len(silent) == 24, "no-read datagram is header plus 8 bytes")

        try:
            decode_binary_datagram(b"\x00" * 24)
            check(False, "bad magic should fail")
        except BinaryError as exc:
            check("magic" in str(exc).lower(), "bad magic is reported")

        udp_events: queue.Queue = queue.Queue()
        udp = UdpEngine(udp_events)
        udp.start("127.0.0.1", 0)
        udp_status = _wait_until(
            udp_events, lambda kind, payload: kind == "status" and payload["state"] == "listening"
        )
        udp_port = int(str(udp_status["detail"]).rsplit(":", 1)[1])
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(packet, ("127.0.0.1", udp_port))
            udp_frames = _collect_frames(udp_events, 1, timeout=3)
            sender.sendto(b"\x00" * 24, ("127.0.0.1", udp_port))
            bad = _wait_until(udp_events, lambda kind, payload: kind == "log" and "Bad magic" in str(payload["text"]))
        finally:
            sender.close()
            udp.stop()
            _wait_thread(udp)
        check(udp_frames[0]["label"] == "42" and udp_frames[0]["heading"] == 87.5, "udp socket delivers binary")
        check("Bad magic" in str(bad["text"]), "udp rejects a bad datagram")

        app.update_idletasks()
        app._show_frame(sample)
        app.update_idletasks()
        check(app.value_vars["label"].get() == "42", "ui shows label")
        check(app.value_vars["x"].get() == "15.2 mm", "ui shows x")
        check(app.value_vars["confidence"].get() == "14 / 15", "ui shows confidence")
        check("1340 µs" in app.value_vars["data_age"].get(), "ui shows data age")
        check(app.pill.cget("text") == "Detected", "ui marks a detection")
        app._show_frame(zero)
        check(app.pill.cget("text") == "No read", "ui marks no read")
        app._show_frame(decoded, "udp", packet)
        check(app.link_pill.cget("text") == "UDP", "ui marks the udp source")
        check(app.value_vars["y"].get() == "-8.4 mm", "ui shows udp y")
        check(app.value_vars["heading"].get() == "87.5 °", "ui shows udp heading")
        check(app.value_vars["message_count"].get() == "1,024", "ui shows udp sequence")
        check("4E 3D" in app.raw.get("1.0", "end"), "ui shows the binary bytes")
        app._show_frame(silent_msg, "udp", silent)
        check(app.pill.cget("text") == "No read", "ui marks a zeroed udp datagram")
        check(app.value_vars["label"].get() == "—", "ui hides an omitted label")
        app.mode_var.set("Listen")
        app._on_mode_changed()
        check(app.address_var.get() == "0.0.0.0:2112", "listen mode fills a bind address")
        check(app.udp_port_var.get() == "5000", "udp port defaults to 5000")
    finally:
        app.destroy()

    print(f"Self-test passed ({len(checks)} checks).")
    return 0


def _wait_until(events: queue.Queue, predicate: Callable[[str, dict], bool], timeout: float = 3) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            kind, payload = events.get(timeout=0.1)
        except queue.Empty:
            continue
        if predicate(kind, payload):
            return payload
    raise TimeoutError("timed out waiting for an engine event")


def _collect_frames(events: queue.Queue, count: int, timeout: float) -> list[dict]:
    found: list[dict] = []
    deadline = time.monotonic() + timeout
    while len(found) < count and time.monotonic() < deadline:
        try:
            kind, payload = events.get(timeout=0.1)
        except queue.Empty:
            continue
        if kind == "frame":
            message = payload["message"]
            assert isinstance(message, dict)
            found.append(message)
    if len(found) < count:
        raise TimeoutError(f"expected {count} frames, got {len(found)}")
    return found


def _wait_thread(engine: ReceiverEngine | UdpEngine) -> None:
    thread = engine._thread
    if thread:
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Receive TCP JSON and UDP binary estimates.")
    parser.add_argument("--self-test", action="store_true", help="check framing and the window, then exit")
    args = parser.parse_args()
    if args.self_test:
        try:
            return self_test()
        except Exception as exc:
            print(f"Self-test failed: {exc}", file=sys.stderr)
            return 1
    _enable_windows_dpi()
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
