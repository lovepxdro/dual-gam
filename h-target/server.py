"""
Dual-GAM — h-target: servidor alvo

Servidor HTTP simples que:
- Responde requisições (para o tráfego legítimo dos h1-h4)
- Registra métricas de carga (para observar o impacto dos ataques)
- Expõe /metrics para o SDN monitorar
"""

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("h-target")

app = Flask(__name__)

# Métricas em memória (últimos 60s)
metrics = {
    "requests_total": 0,
    "requests_per_second": deque(maxlen=60),
    "bytes_received": 0,
    "connections_by_ip": defaultdict(int),
    "start_time": time.time(),
}
lock = Lock()
window_start = time.time()
window_count = 0


@app.route("/", methods=["GET", "POST", "PUT"])
def index():
    global window_start, window_count
    src_ip = request.remote_addr
    data_len = len(request.data)

    with lock:
        metrics["requests_total"] += 1
        metrics["bytes_received"] += data_len
        metrics["connections_by_ip"][src_ip] += 1
        window_count += 1

        # Atualizar janela de RPS
        now = time.time()
        if now - window_start >= 1.0:
            metrics["requests_per_second"].append(window_count)
            window_count = 0
            window_start = now

    return "OK", 200


@app.route("/metrics")
def get_metrics():
    """Endpoint de métricas para o SDN/monitor."""
    with lock:
        uptime = time.time() - metrics["start_time"]
        rps_list = list(metrics["requests_per_second"])
        rps_avg = sum(rps_list) / max(len(rps_list), 1)
        rps_max = max(rps_list) if rps_list else 0

        # Top 10 IPs por requisições
        top_ips = sorted(
            metrics["connections_by_ip"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

    return jsonify({
        "uptime_seconds": round(uptime, 1),
        "requests_total": metrics["requests_total"],
        "bytes_received": metrics["bytes_received"],
        "rps_avg_60s": round(rps_avg, 2),
        "rps_max_60s": rps_max,
        "unique_ips": len(metrics["connections_by_ip"]),
        "top_ips": dict(top_ips),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": time.time()})


if __name__ == "__main__":
    logger.info("h-target iniciado na porta 80")
    app.run(host="0.0.0.0", port=80, threaded=True)
