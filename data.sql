-- Reset tables
DROP TABLE IF EXISTS visits;
DROP TABLE IF EXISTS users;

-- Users table
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_visit TEXT NOT NULL,
    last_purchase TEXT NOT NULL,
    total_spend REAL NOT NULL
);

-- Visits table
CREATE TABLE visits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    visit_time TEXT NOT NULL,
    purchase TEXT NOT NULL,
    spend REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- Seed users
INSERT INTO users (id, created_at, last_visit, last_purchase, total_spend) VALUES
    ('ID-abc123', '2024-01-01T10:00:00', '2024-02-15T11:00:00', 'Milk', 320.0);
INSERT INTO users (id, created_at, last_visit, last_purchase, total_spend) VALUES
    ('ID-def456', '2024-01-03T09:30:00', '2024-02-10T15:45:00', 'Bread', 250.0);
INSERT INTO users (id, created_at, last_visit, last_purchase, total_spend) VALUES
    ('ID-ghi789', '2024-01-05T13:20:00', '2024-02-12T18:30:00', 'Coffee', 410.0);
INSERT INTO users (id, created_at, last_visit, last_purchase, total_spend) VALUES
    ('ID-jkl012', '2024-01-08T12:10:00', '2024-02-14T17:20:00', 'Tea', 180.0);
INSERT INTO users (id, created_at, last_visit, last_purchase, total_spend) VALUES
    ('ID-mno345', '2024-01-10T16:50:00', '2024-02-16T14:55:00', 'Cheese', 520.0);

-- Seed visits
INSERT INTO visits (user_id, visit_time, purchase, spend, source) VALUES
    ('ID-abc123', '2024-02-15T11:00:00', 'Milk', 80.0, 'seed');
INSERT INTO visits (user_id, visit_time, purchase, spend, source) VALUES
    ('ID-def456', '2024-02-10T15:45:00', 'Bread', 60.0, 'seed');
INSERT INTO visits (user_id, visit_time, purchase, spend, source) VALUES
    ('ID-ghi789', '2024-02-12T18:30:00', 'Coffee', 90.0, 'seed');
INSERT INTO visits (user_id, visit_time, purchase, spend, source) VALUES
    ('ID-jkl012', '2024-02-14T17:20:00', 'Tea', 40.0, 'seed');
INSERT INTO visits (user_id, visit_time, purchase, spend, source) VALUES
    ('ID-mno345', '2024-02-16T14:55:00', 'Cheese', 120.0, 'seed');