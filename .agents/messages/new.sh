#!/usr/bin/env bash
set -euo pipefail

# Script directory
dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build filename from timestamp and optional argument
timestamp=$(date +%Y%m%d_%H%M%S)
if [[ $# -gt 0 ]]; then
    name="${timestamp}.${1}.md"
else
    name="${timestamp}.md"
fi

# Create and report
touch "${dir}/${name}"
echo "${dir}/${name}"
