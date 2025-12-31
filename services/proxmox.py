from collectors.proxmox_collector import _get as pve_get

def list_vms():
    return pve_get("/cluster/resources?type=vm") or []

def list_nodes():
    return pve_get("/cluster/resources?type=node") or []

def get_node_uptime(node: str) -> int:
    st = pve_get(f"/nodes/{node}/status") or {}
    return int(st.get("uptime") or 0)

def list_tasks():
    return pve_get("/cluster/tasks") or []
