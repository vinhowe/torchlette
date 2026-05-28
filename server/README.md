# DiLoCo Relay

WebSocket relay for the DiLoCo distributed-training protocol. Routes
envelope-tagged frames (broadcast / peer / cluster / heads) between
agent peers; the server itself does no training.

## Layout

```
server/
  diloco-server-v2.cjs              # the actual server (v2 wire protocol)
  diloco-server.cjs                 # legacy v1, kept for reference
  systemd/diloco-relay.service      # production unit file
  README.md                         # this
```

## Production deployment (Hetzner)

Sources of truth live at `/opt/diloco/diloco-server.cjs` on the box.
Managed by systemd as `diloco-relay.service` — bound to :443 via
`CAP_NET_BIND_SERVICE` so it doesn't need to run as root.

To redeploy a fresh build:

```bash
scp server/diloco-server-v2.cjs root@5.78.181.14:/opt/diloco/diloco-server.cjs
ssh root@5.78.181.14 "systemctl restart diloco-relay"
```

To inspect:

```bash
ssh root@5.78.181.14 "systemctl status diloco-relay --no-pager"
ssh root@5.78.181.14 "journalctl -u diloco-relay -n 50 --no-pager"
```

## Wire protocol

Client envelopes carry a `target` (`broadcast` / `peer` / `cluster` /
`heads`) and a protocol `msg`. The server reads only the envelope and
forwards the original frame to recipients — it never parses tensor
payloads. See `src/distributed/protocol/messages.ts` for shape and
`src/distributed/transports/websocket-relay.ts` for serdes.
