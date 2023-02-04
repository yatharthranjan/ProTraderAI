import csv


class AccountantData:
    def __init__(self, iteration_id, ticker, time, decision_node, event: [str, None] = None):
        self.iteration_id = iteration_id
        self.ticker = ticker
        self.time = time
        self.decision_node = decision_node
        self.event = event

class Accountant:

    def __init__(self, database):
        self.database = database

    def log(self, accountant_data: AccountantData):
        self.database.insert_trade(accountant_data)

    def load(self):
        return self.database.get_trades()

class Database:
    def insert_trade(self, accountant_data: AccountantData):
        pass

    def get_trades(self):
        pass

class SQLiteDatabase(Database):
    def __init__(self, database_file):
        import sqlite3
        self.conn = sqlite3.connect(database_file)
        self.conn.execute("CREATE TABLE IF NOT EXISTS trades (iteration_id INT, ticker TEXT, time TEXT, "
                          "decision_node TEXT, event TEXT)")

    def insert_trade(self, accountant_data: AccountantData):
        c = self.conn.cursor()
        c.execute("INSERT INTO trades VALUES (?,?,?,?,?)",
                  (accountant_data.iteration_id, accountant_data.ticker, accountant_data.time,
                   accountant_data.decision_node, accountant_data.event))
        self.conn.commit()

    def get_trades(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM trades")
        return c.fetchall()

class FileDatabase(Database):
    def __init__(self, file_name):
        self.file_name = file_name

    def insert_trade(self, accountant_data: AccountantData):
        with open(self.file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerow([accountant_data.iteration_id, accountant_data.ticker, accountant_data.time,
                             accountant_data.decision_node, accountant_data.event])

    def get_trades(self):
        with open(self.file_name, "r") as f:
            reader = csv.reader(f)
            return list(reader)
