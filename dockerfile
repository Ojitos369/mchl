FROM python:3.11

# ENVS
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Mexico_City
ENV LANG=es_MX.UTF-8
ENV LANGUAGE=es_MX:es
ENV LC_ALL=es_MX.UTF-8
ENV APPHOME=/usr/src/app
ENV PYTHONUNBUFFERED 1

# DEPENDENCIES
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
    curl git zsh neovim wget unzip cron locales tzdata curl supervisor htop

# ZSH 
RUN apt install git neovim curl zsh -y
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN chsh -s $(which zsh)

# LOCALES
RUN echo "es_MX.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen es_MX.UTF-8 && \
    ln -snf /usr/share/zoneinfo/America/Mexico_City /etc/localtime && \
    echo "America/Mexico_City" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

# CRONS
RUN mkdir /usr/src/logs
RUN echo "" > /etc/cron.d/my_cron \
    && chmod 0644 /etc/cron.d/my_cron
# RUN echo "* * * * * /usr/src/app/crons/test.sh >> /usr/src/logs/test.log 2>&1" >> /etc/cron.d/my_cron
RUN crontab /etc/cron.d/my_cron

# SUPERVISOR
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord"]

# PYTHON
WORKDIR $APPHOME
COPY . $APPHOME
RUN pip install -r requirements.txt

# docker start mchl-py
# docker rm -f mchl-py && docker image rm mchl-web && docker network rm mchl-net && docker compose up -d
# docker rm -f mchl-py && docker image rm mchl-web && docker compose up -d

