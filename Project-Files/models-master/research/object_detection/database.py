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
    def query(self, action, params):
        print(action)
        cur = (self.connection).cursor()
        cur.execute(action, params)
        (self.connection).commit()
        cur.close()
        (self.connection).close()
    def check(self, action):
        cur = (self.connection).cursor()
        cur.execute(action)
        rows = cur.fetchall()
        for row in rows:
            print(row)
        (self.connection).close()

if __name__ == '__main__':
    db = Database()
    action = "SELECT * FROM record"
    db.check(action)
