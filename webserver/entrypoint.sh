#!/bin/bash

if [ "$DATABASE" = "postgres" ]
then
    echo "Waiting for postgres..."

    while ! nc -z $SQL_HOST $SQL_PORT; do
        sleep 0.1
    done

    echo "postgres started"
fi

# why would you flush time?
# python manage.py flush --no-input
# python manage.py migrate

exec "$@" # all the parameters passed to script
