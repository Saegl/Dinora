#!/usr/bin/env bash

# Needed for NixOS with nixld to work with Nvidia GPU in pytorch
# Most distros don't need this line
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$NIX_LD_LIBRARY_PATH

.venv/bin/python -m dinora
