#!/usr/bin/env bash
set -e
uvicorn server:app --reload --port 8001
