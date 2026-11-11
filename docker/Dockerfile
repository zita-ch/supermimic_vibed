# SuperMimic runtime: Isaac Lab 3 (main) kit-less image + tmux/ffmpeg/compiler.
FROM isaac-lab-kitless:main-46d6cc3
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tmux ffmpeg util-linux build-essential python3-dev && rm -rf /var/lib/apt/lists/*
