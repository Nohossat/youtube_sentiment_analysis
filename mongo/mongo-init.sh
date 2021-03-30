#!/bin/sh

mongo --eval "db.createUser({ user: '$MONGO_INITDB_ROOT_USERNAME', pwd: '$MONGO_INITDB_ROOT_PASSWORD', roles: [{ role: 'readWrite', db: '$MONGO_INITDB_DATABASE' }] });"
mongoimport --db=$DB_NAME --collection=comments --type=csv --headerline --authenticationDatabase $MONGO_INITDB_DATABASE -u $MONGO_INITDB_ROOT_USERNAME -p $MONGO_INITDB_ROOT_PASSWORD --file=comments.csv
