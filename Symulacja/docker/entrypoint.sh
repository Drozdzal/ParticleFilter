#!/usr/bin/env bash

IP=$(cut -d: -f1 <(echo $ROBOCUP_SIMULATOR_ADDR))
PORT=$(cut -d: -f2 <(echo $ROBOCUP_SIMULATOR_ADDR))

while true; do
  /controllers/Melman --webots --webots.address "$IP" --webots.port "$PORT" --moduleChain Melman.py
  sleep 1
done
