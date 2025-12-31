# Portal de Capacity Proxmox — Guia de Implementação (Linux)

Este documento descreve o passo a passo completo para implantar o **Portal de Capacity Proxmox** em um servidor Linux, incluindo dependências, banco de dados, coleta automática, serviço e validações.


## Recursos 

- Coleta horária automática

- Portal rodando em venv

- Banco SQLite com histórico

- Dash + Plotly 

- Apache como proxy (sem conflito)

<br>

##  Visão Geral da Arquitetura

```cat
Proxmox VE
↓ (API JSON)

Coletor Python (cron – 1x/hora)
↓
SQLite (capacity.db)
↓
Dash / Plotly (portal)
↓
Apache (proxy reverso)
↓
Usuários (intranet)
```

## Estrutura do Projeto 

```cash 
/portal_capacity
├── app/ # Dashboard (Dash)
│ └── dashboard.py
├── collectors/ # Coletas
│ ├── proxmox_collector.py
│ └── run_snapshot.sh
├── config/
│ └── settings.py
├── database/
│ └── schema.sql
├── logs/
│ └── collector.log
├── run/
│ └── collector.lock
├── desenvolvimento/ # Virtualenv Python
├── capacity.db # Banco SQLite
├── .env # Credenciais (Proxmox)
└── main.py # Entry point do portal
```

## Pré-requisitos do Sistema

- Linux (Ubuntu / Debian recomendado)
- Acesso ao Proxmox VE
- DNS interno (opcional, mas recomendado)
- Apache já instalado (proxy reverso)
- Python 3.9+

## Instalação de Dependências do Sistema 

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip \
  sqlite3 \
  apache2 \
  cron
  ```
  ## Criar Dirétório do Projeto 
  
```bash
sudo mkdir -p /portal_capacity
sudo chown -R root:root /portal_capacity
cd /portal_capacity
```
Copie os arquivos do projeto para este diretório.

## Criar Virtualenv Python 

```bash 
python3 -m venv /portal_capacity/desenvolvimento
source /portal_capacity/desenvolvimento/bin/activate
```
Instalar dependências Python:

```bash 
pip install --upgrade pip
pip install \
  dash \
  dash-bootstrap-components \
  plotly \
  pandas \
  requests \
  python-dotenv \
  openpyxl
```

## Configurar Variáveis de Ambiente (.env)

Criar o arquivo: 

```bash 
Configurar Variáveis de Ambiente (.env)
```

Conteúdo: 

```bash
PROVIDER_NAME=Proxmox
PROXMOX_HOST=host.proxmox.local
PROXMOX_USER=root@pam
# método token (recomendado)
PROXMOX_TOKEN_ID=portal
PROXMOX_TOKEN_SECRET=SEU_TOKEN
# método senha (alternativa)
# PROXMOX_PASS=SENHA
PROXMOX_VERIFY_SSL=false
```
<strong>NUNCA VERSIONAR ESSE ARQUIVO!!! </strong>

## Banco de Dados (SQLite)

<strong> Criar banco e tabela </strong>

```bash 
sqlite3 /portal_capacity/capacity.db
```

```sql
CREATE TABLE IF NOT EXISTS cluster_capacity (
    ts TEXT,
    cluster_name TEXT,
    cpu_used_ghz REAL,
    cpu_total_ghz REAL,
    mem_used_gb REAL,
    mem_total_gb REAL,
    storage_used_tb REAL,
    storage_total_tb REAL
);

CREATE INDEX IF NOT EXISTS idx_cluster_ts
ON cluster_capacity(cluster_name, ts);

.quit
```

## Testar Coleta Manual

```bash 
cd /portal_capacity
/desenvolvimento/bin/python3 -c \
"from collectors.proxmox_collector import save_snapshot; save_snapshot()"
```

Validar no banco:

```bash
sqlite3 capacity.db "SELECT MAX(ts) FROM cluster_capacity;"
```

## Script de Coleta Horária (Wrapper)

Criar: 

```bash
nano /portal_capacity/run/run_snapshot.sh
```

```sh 
#!/usr/bin/env bash
set -euo pipefail

BASE="/portal_capacity"
VENV_PY="$BASE/desenvolvimento/bin/python3"
LOG="$BASE/logs/collector.log"
LOCK="$BASE/run/collector.lock"

cd "$BASE"

/usr/bin/flock -n "$LOCK" \
  "$VENV_PY" -c "from collectors.proxmox_collector import save_snapshot; save_snapshot()" \
  >> "$LOG" 2>&1

```

Permissões:

```bash
chmod +x /portal_capacity/run/run_snapshot.sh
mkdir -p /portal_capacity/logs /portal_capacity/run
```

## Agendar Coleta no Cron (1x por hora)

```bash 
sudo crontab -e
```

Adicionar:

```bash 
0 * * * * /portal_capacity/run/run_snapshot.sh
```

## Validação da Coleta

```bash 
# Última coleta
sqlite3 capacity.db "SELECT MAX(ts) FROM cluster_capacity;"

# Linhas de hoje
sqlite3 capacity.db \
"SELECT COUNT(*) FROM cluster_capacity WHERE date(ts)=date('now','localtime');"

# Log
tail -n 50 /portal_capacity/logs/collector.log
```

## Healthcheck Manual

```bash 
sqlite3 capacity.db "
SELECT
  CASE
    WHEN (strftime('%s','now','localtime') -
          strftime('%s', replace(substr(MAX(ts),1,19),'T',' '))) > 4200
    THEN 'ERRO: coleta atrasada'
    ELSE 'OK: coleta em dia'
  END
FROM cluster_capacity;"
```

## Serviço systemd do Portal

Criar: 

```bash 
nano /etc/systemd/system/portal_capacity.service
```

```ini
[Unit]
Description=Portal de Capacity Proxmox
After=network.target

[Service]
# Caminho do Python dentro do ambiente virtual
ExecStart=/portal_capacity/desenvolvimento/bin/python /portal_capacity/main.py

EnvironmentFile=/portal_capacity/.env

# Diretório onde o serviço deve rodar
WorkingDirectory=/portal_capacity

# Variáveis de ambiente (se quiser garantir carregamento)
Environment="PYTHONUNBUFFERED=1"

# Habilitar restart automático
Restart=always
RestartSec=5

# Usuário que irá rodar o serviço
User=root
Group=root

# Permitir que o serviço abra porta TCP
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
```

Ativar: 

```bash 
systemctl daemon-reload
systemctl enable --now portal_capacity
systemctl status portal_capacity
```
O Dash escuta em: 

```bash 
http://127.0.0.1:8050
```

## Apache como Proxy Reverso (Intranet)

Habilitar módulos: 

```bash 
a2enmod proxy proxy_http
systemctl reload apache2
```
Criar VirtualHost:

```bash
nano /etc/apache2/sites-available/portalcapacity.conf
```

```apache
<VirtualHost *:80>
    ServerName portalcapacity.t-systems

    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8050/
    ProxyPassReverse / http://127.0.0.1:8050/

    ErrorLog ${APACHE_LOG_DIR}/portalcapacity_error.log
    CustomLog ${APACHE_LOG_DIR}/portalcapacity_access.log combined
</VirtualHost>
```
Ativar: 

```bash
a2ensite portalcapacity
apachectl configtest
systemctl reload apache2
```

## DNS Interno 

Criar registro: 

```bash 
portalcapacity.empresa   A  IP_DO_SERVIDOR
```

Acesso final: 

```bash
http://portalcapacity.empresa 
```
