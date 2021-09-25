FROM ubuntu:21.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake make gcc g++ libarmadillo-dev libyaml-cpp-dev libprotobuf-dev libjsoncpp-dev libasound2-dev protobuf-compiler libboost1.74-dev libfftw3-dev libturbojpeg0-dev libx11-dev libgl-dev libusb-1.0-0-dev libpng-dev libglu1-mesa-dev libgtest-dev libboost-system1.74-dev \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir /home/nobody \
 && chown nobody:users /home/nobody \
 && usermod --home /home/nobody nobody

USER nobody

COPY --chown=nobody:users entrypoint.sh /home/nobody
COPY --chown=nobody:users controllers /home/nobody/controllers

WORKDIR /home/nobody

ENTRYPOINT /home/nobody/entrypoint.sh
