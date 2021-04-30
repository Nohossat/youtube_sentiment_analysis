import random
import string
import sqlite3
import os

import nohossat_cas_pratique

module_path = os.path.dirname(os.path.dirname(os.path.dirname(nohossat_cas_pratique.__file__)))
data_path = os.path.join(module_path, "data", "users.db")

def create_random_pwd():
    letters = ''.join(random.choice(string.ascii_letters) for i in range(10))
    digits = ''.join(random.choice(string.digits) for i in range(10))
    random_sequence = letters + digits
    pwd = ''.join(random.choice(random_sequence) for i in range(14))

    return pwd


def create_connection(action, db_file=data_path):
  conn = None
  res = None
  try:
    conn = sqlite3.connect(db_file)
    res = action(conn)
  except Exception as e:
    print(e)
  finally:
    if conn:
      conn.close()
    return res


def save_database(user, email, pwd):

    def insert_new_user(conn):
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", (user, email, pwd))
        conn.commit()

    create_connection(insert_new_user)


def get_existing_user(user):
    def get_user(conn):
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE name = '%s'" % user)
        res = c.fetchone()
        return res

    return create_connection(get_user)


def delete_user(user):
    def get_user(conn):
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE name = '%s'" % user)
        conn.commit()

    return create_connection(get_user)
