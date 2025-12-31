import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "capacity.db")

PROVIDER_NAME = os.getenv("PROVIDER_NAME", "Proxmox")

PROXMOX_HOST = os.getenv("PROXMOX_HOST")
PROXMOX_USER = os.getenv("PROXMOX_USER")
PROXMOX_TOKEN_ID = os.getenv("PROXMOX_TOKEN_ID")
PROXMOX_TOKEN_SECRET = os.getenv("PROXMOX_TOKEN_SECRET")
PROXMOX_PASS = os.getenv("PROXMOX_PASS")
PROXMOX_VERIFY_SSL = os.getenv("PROXMOX_VERIFY_SSL", "false").strip().lower() in ("1", "true", "yes")
