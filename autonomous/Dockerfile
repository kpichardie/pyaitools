from python:3.14.2
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    apt-get update && \
    apt-get install -y --no-install-recommends debian-archive-keyring
Run apt-get upgrade -y
RUN pip install --upgrade pip --no-cache-dir --progress-bar off
Run pip install --no-cache-dir --progress-bar off google-genai
Run pip install --no-cache-dir --progress-bar off python-telegram-bot python-telegram-bot[job-queue]
