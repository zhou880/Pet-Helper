import psycopg2
from secrets import *

class Database:
    def __init__(self):
        self.connection = psycopg2.connect(
            host = HOST,
            database = DATABASE,
            user = USER,
            password = PASSWORD,
            port = PORT
        )
    def query(self, action):
        cur = (self.connection).cursor()
        cur.execute(action)
        rows = cur.fetchall()
        print(rows)
        (self.connection).close()

if __name__ == '__main__':
    db = Database()
    action = "SELECT first_name FROM actor LIMIT 10"
    db.query(action)
