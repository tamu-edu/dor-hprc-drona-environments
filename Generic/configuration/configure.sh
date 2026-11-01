#!/bin/bash
source /etc/profile

# cd to the env conf dirdir
cd $DRONA_ENV_DIR

configuration/generate_modules_db.py -b

mkdir -p modules
mv configuration/modules.sqlite3 modules

echo "Module database created in directory $DRONA_ENV_DIR/modules"

echo

cat configuration/README


touch $DRONA_ENV_DIR/configuration/configured_check
