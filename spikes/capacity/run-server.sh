#!/usr/bin/env bash
# Run ON THE SERVER from the staged source directory. Never calls SSH.
set -euo pipefail
IP_STEP08_RUN="${2:?fresh run ID required}"
[[ "$IP_STEP08_RUN" =~ ^[a-z0-9-]+$ ]] || exit 2
IP_STEP08_ROOT="/datasets/byl/insightpilot/step08/$IP_STEP08_RUN"
IP_STEP08_SOURCE="$IP_STEP08_ROOT/source"
IP_STEP08_CACHE=/datasets/byl/insightpilot/step02/cache
IP_STEP08_BASE=sha256:c33eededb937da602a32184e47a35e35c4bc4373cd690c1a22fd6cf7d5001eae
IP_STEP08_GPU=GPU-ec35358d-4afa-1765-2f19-5d2fe68ede13
IP_STEP08_PRECISION="${1:?fp16 or fp32}"
IP_STEP08_HEADROOM="${3:-4294967296}"
[[ "$IP_STEP08_HEADROOM" =~ ^[0-9]+$ ]] || exit 2
case "$IP_STEP08_PRECISION" in fp16|fp32) ;; *) exit 2 ;; esac
IP_STEP08_PROJECT="insightpilot-model-test-step08-$IP_STEP08_RUN-$IP_STEP08_PRECISION"
IP_STEP08_NAME="$IP_STEP08_PROJECT-probe"
IP_STEP08_GROUP=$(stat -c %g /var/run/docker.sock)
test -f "$IP_STEP08_ROOT/.env.capacity-model"
test -d "$IP_STEP08_CACHE"
# BuildKit interprets a bare sha256 ID in FROM as a registry repository name.
# Give the verified existing image a project-owned local tag first.
docker image tag "$IP_STEP08_BASE" insightpilot-step08-base:c33eededb937
timeout 600 docker build --pull=false --build-arg PROBE_BASE=insightpilot-step08-base:c33eededb937 \
  --file "$IP_STEP08_SOURCE/spikes/capacity/Dockerfile.remote" \
  --tag insightpilot-step08-model:local "$IP_STEP08_SOURCE"
IP_STEP08_IMAGE=$(docker image inspect insightpilot-step08-model:local --format '{{.Id}}')
# This is a one-shot attempt: no automatic retry, CPU fallback or OOM microbatch repair.
# Pass a third argument of 0 only for an explicitly authorized reserve exception.
if ss -H -ltn 'sport = :18108' | read -r _; then
  ss -ltnp 'sport = :18108'
  exit 1
fi
mkdir -p "$IP_STEP08_ROOT/evidence"
nvidia-smi > "$IP_STEP08_ROOT/evidence/$IP_STEP08_PRECISION-before-nvidia.txt"
docker ps -a --format '{{.ID}} {{.Names}} {{.State}}' > "$IP_STEP08_ROOT/evidence/$IP_STEP08_PRECISION-before-containers.txt"
docker run -d --name "$IP_STEP08_PROJECT-observer" \
  --label com.docker.compose.project=insightpilot-model-test-step08-monitor \
  --read-only --user "$(id -u):$(id -g)" --group-add "$IP_STEP08_GROUP" \
  --network none --memory 256m --cpus 0.5 --pids-limit 128 \
  --gpus "device=$IP_STEP08_GPU" \
  --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  --mount type=bind,src=/usr/bin/docker,dst=/usr/bin/docker,readonly \
  --mount "type=bind,src=$IP_STEP08_SOURCE,dst=/opt/probe,readonly" \
  --mount "type=bind,src=$IP_STEP08_ROOT/evidence,dst=/evidence" \
  --workdir /opt/probe \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "IP_CAPACITY__PROJECT=$IP_STEP08_PROJECT" \
  --env "IP_CAPACITY__OUTPUT=/evidence/server-$IP_STEP08_PRECISION-observation.json" \
  --env "IP_CAPACITY__STOP_FILE=/evidence/$IP_STEP08_PRECISION-observation.stop" \
  --env IP_CAPACITY__DURATION_S=1200 --env IP_CAPACITY__INTERVAL_S=1 \
  --env "IP_CAPACITY__GPU_ID=$IP_STEP08_GPU" \
  --env "IP_CAPACITY__GPU_HEADROOM_BYTES=$IP_STEP08_HEADROOM" \
  --env "IP_CAPACITY__READER_IMAGE=$IP_STEP08_IMAGE" \
  "$IP_STEP08_IMAGE" python -m spikes.capacity.observe
docker run -d --name "$IP_STEP08_NAME" \
  --label "com.docker.compose.project=$IP_STEP08_PROJECT" \
  --read-only --user 10001:10001 --cpus 4 --memory 12g --pids-limit 256 --cgroupns private \
  --gpus "device=$IP_STEP08_GPU" --publish 127.0.0.1:18108:8100 \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --mount "type=bind,src=$IP_STEP08_CACHE,dst=/models,readonly" \
  --mount "type=bind,src=$IP_STEP08_SOURCE,dst=/opt/probe,readonly" \
  --workdir /opt/probe --env-file "$IP_STEP08_ROOT/.env.capacity-model" \
  --env "IP_MODEL_SERVER__PRECISION=$IP_STEP08_PRECISION" \
  --env "IP_PROBE__GPU_HEADROOM_BYTES=$IP_STEP08_HEADROOM" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_HOME=/models \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
  --log-opt max-size=10m --log-opt max-file=3 \
  "$IP_STEP08_IMAGE" timeout 1200s python -m spikes.capacity.model_probe
