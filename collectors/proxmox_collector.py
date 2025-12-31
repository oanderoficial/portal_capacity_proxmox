import requests
import urllib3
from datetime import datetime

from database.db import get_connection, init_db
from config.settings import (
    PROXMOX_HOST,
    PROXMOX_USER,
    PROXMOX_TOKEN_ID,
    PROXMOX_TOKEN_SECRET,
    PROXMOX_PASS,
    PROXMOX_VERIFY_SSL,
)


def _base_url() -> str:
    if not PROXMOX_HOST:
        raise RuntimeError("PROXMOX_HOST não definido. Verifique seu .env")
    host = PROXMOX_HOST.strip()
    if host.startswith("http://") or host.startswith("https://"):
        base = host.rstrip("/")
        if not base.endswith("/api2/json"):
            base = f"{base}/api2/json"
        return base
    return f"https://{host}:8006/api2/json"


def _session() -> requests.Session:
    s = requests.Session()
    s.verify = PROXMOX_VERIFY_SSL
    if not PROXMOX_VERIFY_SSL:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if PROXMOX_USER and PROXMOX_TOKEN_ID and PROXMOX_TOKEN_SECRET:
        s.headers.update(
            {
                "Authorization": f"PVEAPIToken={PROXMOX_USER}!{PROXMOX_TOKEN_ID}={PROXMOX_TOKEN_SECRET}"
            }
        )
        return s

    if PROXMOX_USER and PROXMOX_PASS:
        resp = s.post(
            f"{_base_url()}/access/ticket",
            data={"username": PROXMOX_USER, "password": PROXMOX_PASS},
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        ticket = data.get("ticket")
        if not ticket:
            raise RuntimeError("Falha ao obter ticket do Proxmox")
        s.headers.update({"Cookie": f"PVEAuthCookie={ticket}"})
        return s

    raise RuntimeError("Credenciais do Proxmox não definidas corretamente")


def _get(path: str):
    s = _session()
    resp = s.get(f"{_base_url()}{path}")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 403:
            raise RuntimeError("Proxmox: acesso negado (403). Conceda permissão de auditoria (PVEAuditor) no '/' ao usuário/token.") from e
        raise
    return resp.json().get("data")


def _get_cluster_name() -> str:
    status = _get("/cluster/status")
    if isinstance(status, list):
        for item in status:
            if item.get("type") == "cluster" and item.get("name"):
                return str(item["name"])
    return "Proxmox"


def _get_nodes_metrics():
    nodes = _get("/cluster/resources?type=node") or []
    total_cpu_cores = 0.0
    used_cpu_cores = 0.0
    total_mem_bytes = 0
    used_mem_bytes = 0

    for n in nodes:
        maxcpu = float(n.get("maxcpu") or 0.0)
        cpu_ratio = float(n.get("cpu") or 0.0)
        maxmem = int(n.get("maxmem") or 0)
        mem = int(n.get("mem") or 0)

        total_cpu_cores += maxcpu
        used_cpu_cores += cpu_ratio * maxcpu
        total_mem_bytes += maxmem
        used_mem_bytes += mem

    return total_cpu_cores, used_cpu_cores, total_mem_bytes, used_mem_bytes


def _get_storage_metrics():
    storages = _get("/cluster/resources?type=storage") or []
    total_bytes = 0
    used_bytes = 0

    seen_ids = set()
    for s in storages:
        sid = s.get("storage") or s.get("id")
        if sid and sid in seen_ids:
            continue
        # Proxmox usa 'maxdisk' (total em bytes) e 'disk' (uso em bytes)
        total = int(s.get("maxdisk") or 0)
        used = int(s.get("disk") or 0)
        if total <= 0:
            # Alguns storages podem reportar 0 em cluster/resources.
            # Tenta endpoint de status por storage se possível (opcional).
            # Mantemos 0 para não travar a coleta.
            pass
        total_bytes += total
        used_bytes += used
        if sid:
            seen_ids.add(sid)

    return total_bytes, used_bytes


def get_cluster_usage():
    cluster_name = _get_cluster_name()

    cpu_total_cores, cpu_used_cores, mem_total_b, mem_used_b = _get_nodes_metrics()
    sto_total_b, sto_used_b = _get_storage_metrics()

    cpu_total_ghz = float(cpu_total_cores)
    cpu_used_ghz = float(cpu_used_cores)
    mem_total_gb = float(mem_total_b) / (1024 ** 3)
    mem_used_gb = float(mem_used_b) / (1024 ** 3)
    storage_total_tb = float(sto_total_b) / (1024 ** 4)
    storage_used_tb = float(sto_used_b) / (1024 ** 4)

    return [
        {
            "cluster": cluster_name,
            "cpu_total": cpu_total_ghz,
            "cpu_used": cpu_used_ghz,
            "mem_total": mem_total_gb,
            "mem_used": mem_used_gb,
            "storage_total": storage_total_tb,
            "storage_used": storage_used_tb,
        }
    ]


def save_snapshot():
    init_db()
    data = get_cluster_usage()
    ts = datetime.utcnow().isoformat()

    conn = get_connection()
    cur = conn.cursor()

    for c in data:
        cur.execute(
            """
            INSERT INTO cluster_capacity (
                ts, cluster_name,
                cpu_used_ghz, cpu_total_ghz,
                mem_used_gb, mem_total_gb,
                storage_used_tb, storage_total_tb
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ts,
                c["cluster"],
                c["cpu_used"],
                c["cpu_total"],
                c["mem_used"],
                c["mem_total"],
                c["storage_used"],
                c["storage_total"],
            ),
        )

    conn.commit()
    conn.close()
    print(f"Snapshot salvo em {ts}")
