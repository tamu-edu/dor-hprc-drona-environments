#!/usr/bin/env python3
import os
import sqlite3
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool
import argparse
import time

CWD_PATH = os.path.dirname(os.path.realpath(__file__))
MAX_PROCESSES=6

def load_db(db_file_path):
    conn = sqlite3.connect(db_file_path)
    return conn

def insert_module(cursor, module_name, description):
     # Insert a row of data
    cursor.execute("INSERT INTO modules VALUES (?,?)", (module_name, description))

# source: https://docs.python.org/2.7/library/sqlite3.html
def create_modules_table(conn):
    c = conn.cursor()

    # drop table if exists
    c.execute('drop table if exists modules')

    # Create table
    c.execute('''CREATE TABLE modules
                (name text PRIMARY KEY, description text)''')

    # Save (commit) the changes
    conn.commit()

    return conn

def insert_modules(conn, module_dict):
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO modules VALUES (?,?)", module_dict)
    conn.commit()
    cursor.close()

    return conn

def create_db_from_list(db_file_path, module_dict):
    # create a new database from scratch
    conn = load_db(db_file_path)
    conn = create_modules_table(conn)

    insert_modules(conn, module_dict)

    return conn

def read_avail_modules(toolchain=None):
    lmod_path = os.getenv('LMOD_CMD')

    if lmod_path is None:
        return []
    
    if toolchain:
        # Shell command to get the currently loaded modules from that toolchain
        shell_command = f"""
        module purge
        module load {toolchain}
        {lmod_path} bash -t avail
        """
    else:
        shell_command = f"{lmod_path} bash -t avail"
    
    p = Popen(shell_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout_data =  p.communicate(input='q'.encode())[0].decode('utf-8')

    delimiter = "MODULEPATH" # this is where the junk starts
    delimiter_pos = stdout_data.find(delimiter)
    # does not follow expected format, return empty
    if delimiter_pos == -1:
        return []

    # remove the first line (header)
    stdout_data = stdout_data[:delimiter_pos]
    stdout_data = stdout_data.split('\n')[1:] # remove LMOD internal string
    stdout_data = [module_name for module_name in stdout_data if not module_name.endswith('/')] # filter out directory

    module_names = [name.strip() for name in stdout_data]
    module_names = frozenset(module_names) # we need to strictly make the module name unique
    return module_names

def read_module_description(name):
    lmod_path = os.getenv('LMOD_CMD')
    if lmod_path is None:
        return ""
    
    p = Popen([lmod_path, "bash", "spider", name], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    stdout_data =  p.communicate(input='q'.encode())[0].decode('utf-8')

    delimiter = "__LMOD_REF_COUNT"
    delimiter_pos = stdout_data.find(delimiter)

    # does not follow expected format, return empty
    if delimiter_pos == -1:
        return ""
    
    module_description = stdout_data[:delimiter_pos]
    module_description = module_description.strip()

    return module_description

def lookup_module(module_name):
    """ Look up a description for the given module name

        :type module_name: string

        :param module_name: the module name to lookup
    
        :rtype: (module name, module description) (both are in str encoding)
    """

    utf8_description = read_module_description(module_name)
    res = (module_name, utf8_description)

    return res  

def modules_in_db(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM modules;")
    modules = cursor.fetchall()
    modules = [row[0] for row in modules ]
    modules_set = frozenset(modules)

    cursor.close()

    return modules_set

def index_modules(conn, module_names):
    num_new_modules = len(module_names)
    print("Adding {0} module(s) to database.".format(num_new_modules))
    p = Pool(MAX_PROCESSES) # expect that the number of new modules will be small
    module_dict = p.map(lookup_module, module_names)

    insert_modules(conn, module_dict)

    return conn

def update_new_modules(conn, avail_modules, indb_modules):
    # new modules = available modules at the time of this run - everything else in the database
    new_modules = avail_modules.difference(indb_modules)
    print("Adding {0}".format(new_modules))

    num_new_modules = len(new_modules)
    if num_new_modules == 0:
        print("No new modules to add.")
        return conn

    index_modules(conn, new_modules)

    return conn


def remove_obsolete_modules(conn, avail_modules, indb_modules):
    # removed any modules that is in the database but not in the available module list 
    removed_modules = indb_modules.difference(avail_modules)
    removed_modules = [(name,) for name in removed_modules]

    # print(removed_modules)
    num_removed_modules = len(removed_modules)
    if num_removed_modules == 0:
        print("No obsolete modules to remove.")
        return conn

    print("Removing {0} module(s) from the database.".format(num_removed_modules))
    cursor = conn.cursor()
    cursor.executemany('''DELETE FROM modules WHERE name=?''', removed_modules)
    conn.commit()

    return conn

def update_db(db_file_path):
    start = time.time()
    conn = load_db(db_file_path)
    avail_modules = read_avail_modules(toolchain)
    indb_modules = modules_in_db(conn)

    
    # check for new modules and add if necessary
    conn = update_new_modules(conn, avail_modules, indb_modules)
    conn = remove_obsolete_modules(conn, avail_modules, indb_modules)

    conn.close()
    end = time.time()
    print("Done updating database. Took {0}".format(end - start))

# this is for testing only
# in case you need to look at the output of module avail -t
def create_module_file(modules_list):
    f = open('test.txt', 'w')
    for name in modules_list:
        f.write('{0}\n'.format(name))
    f.close()

def rebuild_db(db_file_path):
    start = time.time()
    avail_modules = read_avail_modules(toolchain)

    print("Building indices for {0} modules".format(len(avail_modules)))
    p = Pool(MAX_PROCESSES)
    module_dict = p.map(lookup_module, avail_modules)
   
    conn = create_db_from_list(db_file_path, module_dict)
    end = time.time()
    print("Indexing {0} took {1}".format(len(avail_modules), end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Module indexing utilities for fast_ml.")
    parser.add_argument('-b', '--build', action='store_true', help="Build modules database from the ground up with specified toolchain.")
    parser.add_argument('-u', '--update', action='store_true', help="Update the database (add new modules + remove obsolete ones.)")
    parser.add_argument('-t', '--toolchain', help="Specify the toolchain (e.g., intel/2020b)")
    args = parser.parse_args()

    toolchain = args.toolchain if args.toolchain else None

    if toolchain:
        # Renaming the database name for the chosen toolchain
        DATABASE_FILE = os.path.join(CWD_PATH, f"modules-{toolchain.replace('/', '_')}.sqlite3")
    else:
        DATABASE_FILE = os.path.join(CWD_PATH, "modules.sqlite3")

    if args.build:
        rebuild_db(DATABASE_FILE)
        exit(0)
    elif args.update:
        update_db(DATABASE_FILE)
        exit(0)
    else:
        parser.print_help()