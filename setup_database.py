# setup_database.py
import sqlite3
import os

DB_FILE = "main_db.sqlite"

# Delete the old DB file if it exists to start fresh
if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

print("Creating tables: users, products, reviews...")
# Create tables
cursor.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);''')
cursor.execute('''CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);''')
cursor.execute('''CREATE TABLE reviews (id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, rating INTEGER, comment TEXT, FOREIGN KEY(user_id) REFERENCES users(id), FOREIGN KEY(product_id) REFERENCES products(id));''')

print("Inserting sample data...")
# Insert data
cursor.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');")
cursor.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com');")

cursor.execute("INSERT INTO products (name, category, price) VALUES ('Quantum Laptop', 'Electronics', 1200.00);")
cursor.execute("INSERT INTO products (name, category, price) VALUES ('Nebula Smartwatch', 'Wearables', 350.50);")

cursor.execute("INSERT INTO reviews (user_id, product_id, rating, comment) VALUES (1, 1, 5, 'Absolutely fantastic, blazing fast!');")
cursor.execute("INSERT INTO reviews (user_id, product_id, rating, comment) VALUES (2, 1, 4, 'Great machine, but battery could be better.');")
cursor.execute("INSERT INTO reviews (user_id, product_id, rating, comment) VALUES (1, 2, 5, 'Stylish and very functional, love it!');")


conn.commit()
conn.close()
print(f"Database '{DB_FILE}' created and populated successfully.")