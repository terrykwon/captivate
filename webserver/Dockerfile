FROM python:3.7-slim

WORKDIR /usr/src/webserver

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install psycopg2
# debian uses apt
# RUN apt update \
#         && apt-get install postgresql-dev gcc python3-dev musl-dev

# install netcat, used in entrypoint.sh
RUN apt-get update && apt-get install -y \
    netcat \
    libgl1-mesa-dev \
    libgtk2.0-dev

RUN pip install --upgrade pip setuptools wheel
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./entrypoint.sh .

# copy project to container
COPY . .

# run entrypoint, which checks the postgres connection and migrates
ENTRYPOINT ["/usr/src/webserver/entrypoint.sh"]