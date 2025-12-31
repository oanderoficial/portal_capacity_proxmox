# CHECKLIST DE CHECAGEM DA COLETA 

## 1- Ver se o cron está cadastrado (1ª coisa)

```bash 
sudo crontab -l
```

<P> Você precisa ver: </P>

```bash
0 * * * * /portal_capacity/run/run_snapshot.sh
```

## 2 - Ver se o cron executou nesse horário

```bash 
sudo grep run_snapshot.sh /var/log/syslog | tail -n 20
```

<P> Você deve ver algo como:: </P>

```bash 
CRON[xxxxx]: (root) CMD (/portal_capacity/run/run_snapshot.sh)
```


## 3 - Ver o log do coletor (principal evidência)

```bash
tail -n 100 /portal_capacity/logs/collector.log
```

<P> Você deve ver novas linhas com timestamp atual. </P>

## 4️ - Ver a última coleta gravada no banco (o mais importante)

```bash
sqlite3 /portal_capacity/capacity.db "SELECT MAX(ts) FROM cluster_capacity;"
```
<P> ➡️ Esse timestamp precisa estar ~1 hora mais novo do que antes. </P>

## 5️ - Ver quantas linhas foram gravadas HOJE

```bash 
sqlite3 /portal_capacity/capacity.db \
"SELECT COUNT(*) FROM cluster_capacity WHERE date(ts)=date('now','localtime');"
```

<P> ➡️ O número deve aumentar a cada hora. </P>

## 6️ -  Ver as últimas linhas inseridas (confirmação visual)

```bash 
sqlite3 /portal_capacity/capacity.db \
"SELECT ts, cluster_name FROM cluster_capacity ORDER BY ts DESC LIMIT 10;"
```

## 7️ - Healthcheck rápido (OK / ERRO)

```
sqlite3 /portal_capacity/capacity.db "
SELECT
  CASE
    WHEN (strftime('%s','now','localtime') - strftime('%s', replace(substr(MAX(ts),1,19),'T',' '))) > 4200
    THEN 'ERRO: coleta atrasada'
    ELSE 'OK: coleta em dia'
  END
FROM cluster_capacity;"
```

## 8️ - Ver se o cron está ativo (geral)

```bash 
systemctl status cron
```

## 9️ - Ver cron rodando em tempo real

```bash
sudo tail -f /var/log/syslog | grep CRON
```

## Ordem recomendada quando for checar

1) SELECT MAX(ts)

2) tail collector.log

3) grep run_snapshot.sh /var/log/syslog


Se esses 3 estiverem OK → 100% garantido que a coleta está funcionando.
