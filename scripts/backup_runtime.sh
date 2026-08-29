#!/usr/bin/env bash
set -euo pipefail

container="${TRAE_RELAY_CONTAINER:-trae-cn-relay}"
project_root="${TRAE_RELAY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
timestamp="${1:-$(date +%Y%m%d-%H%M%S)}"
backup_dir="$project_root/backups/${timestamp}-runtime"
image="trae-cn-relay:backup-${timestamp}"

install -d -m 700 "$backup_dir"
was_running="$(docker inspect -f '{{.State.Running}}' "$container")"

restart_container() {
  if [[ "$was_running" == "true" ]]; then
    docker start "$container" >/dev/null || true
  fi
}
trap restart_container EXIT

if [[ "$was_running" == "true" ]]; then
  docker stop -t 30 "$container" >/dev/null
fi

docker commit --pause=false "$container" "$image" >"$backup_dir/IMAGE_ID"
docker cp "$container:/app/src" "$backup_dir/container-src" >/dev/null
tar -C "$backup_dir" -czf "$backup_dir/container-src.tar.gz" container-src
rm -rf "$backup_dir/container-src"

host_items=(src Dockerfile docker-compose.yml requirements.txt web_login.py start_auth.bat)
[[ -f "$project_root/.env" ]] && host_items+=(.env)
[[ -d "$project_root/data" ]] && host_items+=(data)
tar -C "$project_root" -czf "$backup_dir/host-state.tar.gz" "${host_items[@]}"

docker inspect "$container" >"$backup_dir/container.inspect.json"
docker diff "$container" >"$backup_dir/container.diff.txt"

restart_container
trap - EXIT

docker save "$image" | gzip -1 >"$backup_dir/trae-cn-relay-image.tar.gz"
sha256sum "$backup_dir"/*.tar.gz "$backup_dir/IMAGE_ID" >"$backup_dir/SHA256SUMS"
chmod 600 "$backup_dir"/*
sha256sum -c "$backup_dir/SHA256SUMS"

printf 'backup_dir=%s\nimage=%s\n' "$backup_dir" "$image"
