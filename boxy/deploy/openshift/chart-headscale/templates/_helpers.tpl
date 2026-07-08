{{/* The Route/MagicDNS host: explicit route.host, else the serverUrl host. */}}
{{- define "headscale.host" -}}
{{- if .Values.route.host -}}
{{- .Values.route.host -}}
{{- else -}}
{{- .Values.serverUrl | trimPrefix "https://" | trimPrefix "http://" | splitList "/" | first -}}
{{- end -}}
{{- end -}}

{{/* Common labels. */}}
{{- define "headscale.labels" -}}
app: headscale
app.kubernetes.io/name: headscale
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* headscale config.yaml (rendered into the ConfigMap). */}}
{{- define "headscale.config" -}}
server_url: {{ required "serverUrl is required" .Values.serverUrl }}
listen_addr: 0.0.0.0:8080
metrics_listen_addr: 127.0.0.1:9090
grpc_listen_addr: 127.0.0.1:50443
noise:
  private_key_path: /var/lib/headscale/noise_private.key
prefixes:
  v4: 100.64.0.0/10
database:
  type: sqlite
  sqlite:
    path: /var/lib/headscale/db.sqlite
dns:
  magic_dns: {{ .Values.magicDns }}
  base_domain: {{ .Values.baseDomain }}
  nameservers:
    global:
      - 1.1.1.1
derp:
  server:
    enabled: true
    region_id: 999
    region_code: boxy
    stun_listen_addr: 0.0.0.0:3478
  urls: []
  auto_update_enabled: false
{{- end -}}
