import os

from nohossat_cas_pratique.user_creation import create_random_pwd, create_connection, get_existing_user, save_database, delete_user


def test_create_random_pwd():
    pwd = create_random_pwd()
    assert isinstance(pwd, str)


def test_create_connection():
    def get_user(conn):
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE name = "nono_dev";')
        res = c.fetchone()
        return res

    res = create_connection(get_user)

    assert res[1] == "nohossat.traore@gmail.com"


def test_get_existing_user():
    res = get_existing_user("nono_dev")
    assert res[1] == "nohossat.traore@gmail.com"


def test_save_database():
    save_database("test_user_save", "nohossat.traore@gmail.com", "1234")
    res = get_existing_user("test_user_save")

    assert res[0] == "test_user_save"
    assert res[1] == "nohossat.traore@gmail.com"
    delete_user("test_user_save")
    res = get_existing_user("test_user_save")
    assert res == None


def test_save_database():
    save_database("test_user_delete", "nohossat.traore@gmail.com", "1234")
    res = get_existing_user("test_user_delete")
    assert res[0] == "test_user_delete"
    assert res[1] == "nohossat.traore@gmail.com"

    delete_user("test_user_delete")
    res = get_existing_user("test_user_delete")
    assert res == None
