# First Party
from fms_dgt.validators.nl2sql.sql_execution_validator import SQLExecutionValidator
from fms_dgt.validators.nl2sql.sql_syntax_validator import SQLSyntaxValidator


def test_sql_syntax_validator():
    validator = SQLSyntaxValidator(name="sql_syntax_validator", config={})
    assert validator._validate(
        record=dict(
            sql_schema="CREATE TABLE users (\n    user_id INTEGER,\n    first_name VARCHAR(155),\n    last_name VARCHAR(155),\n    email VARCHAR(155),\n    city VARCHAR(155),\n    country VARCHAR(155)\n);\nCREATE TABLE orders (\n    order_id INTEGER,\n    user_id INTEGER,\n    product_name VARCHAR(155),\n    price FLOAT,\n    quantity INT,\n    order_date DATE,\n    order_state VARCHAR(155),\n    CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (user_id)\n);",
            sql_query="SELECT * FROM users",
            utterance="Get me all the users",
        ),
    )
    assert not validator._validate(
        record=dict(
            sql_schema="CREATE TABLE broken_table (column character,",
            sql_query="SELECT * FROM",
            utterance="Get me all the users",
        ),
    )


def test_sql_execution_validator():
    validator = SQLExecutionValidator(name="sql_execution_validator", config={})
    assert validator._validate(
        record=dict(
            sql_schema="CREATE TABLE users (\n    user_id INTEGER,\n    first_name VARCHAR(155),\n    last_name VARCHAR(155),\n    email VARCHAR(155),\n    city VARCHAR(155),\n    country VARCHAR(155)\n);\nCREATE TABLE orders (\n    order_id INTEGER,\n    user_id INTEGER,\n    product_name VARCHAR(155),\n    price FLOAT,\n    quantity INT,\n    order_date DATE,\n    order_state VARCHAR(155),\n    CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (user_id)\n);",
            sql_query="SELECT * FROM users",
            utterance="Get me all the users",
        ),
    )
    assert not validator._validate(
        record=dict(
            sql_schema="CREATE TABLE users (\n    user_id INTEGER,\n    first_name VARCHAR(155),\n    last_name VARCHAR(155),\n    email VARCHAR(155),\n    city VARCHAR(155),\n    country VARCHAR(155)\n);\nCREATE TABLE orders (\n    order_id INTEGER,\n    user_id INTEGER,\n    product_name VARCHAR(155),\n    price FLOAT,\n    quantity INT,\n    order_date DATE,\n    order_state VARCHAR(155),\n    CONSTRAINT orders_user_id_fkey FOREIGN KEY (user_id) REFERENCES users (user_id)\n);",
            sql_query="SELECT * FROM not_existing_table",
            utterance="Get me all the users",
        ),
    )
