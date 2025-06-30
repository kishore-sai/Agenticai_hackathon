import sqlglot as sg
import sqlglot.expressions as exp
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import streamlit as st


def validate_query(sql_query: str, docs: list):
    """Validates a SQL query and returns True or False based on query correctness.

    Args:
        sql_query: The SQL query to validate
        docs: Context Documents generated for user question

    Returns:
        True or False
    """
    # retrieval_docs = [docs for docs, _ in docs]
    try:
        tree = sg.parse_one(sql_query, dialect="snowflake")
    except Exception as e:
        print("SQL query cannot be parsed", e)
        return False, getattr(
            e,
            "message",
            "SQL query is not valid. Check if it conforms to ANSI standards.",
        )

    column_names = set()
    table_names = set()

    for doc in docs:
        column_names.add(doc["column"].strip().lower())
        table_names.add(doc["table"].strip().lower())
        if doc["foreign_key_ref"]:
            x = sg.parse_one(doc["foreign_key_ref"])
            for fk_col in x.find_all(exp.Column):
                column_names.add(fk_col.name.strip().lower())
                table_names.add(fk_col.table.strip().lower())

    if not st.session_state.similar_inputs.empty:
        print("valid")
        for query in st.session_state.similar_inputs["sql_query"]:
            try:
                parsed = sg.parse_one(query, read='tsql')  # Use SQL Server dialect

                # Extract and add table names
                tables = parsed.find_all(sg.expressions.Table)
                for table in tables:
                    table_names.add(table.name.strip().lower())

                # Extract and add column names
                columns = parsed.find_all(sg.expressions.Column)
                for col in columns:
                    column_names.add(col.name.strip().lower())

            except Exception as e:
                print(f"❌ Error parsing SQL query: {e}")
    print(column_names)
    for tbl in tree.find_all(exp.CTE):
        table_names.add(
            tbl.alias_or_name.lower()
        )


    for tbl in tree.find_all(exp.Table):
        print("Checking TABLE NAME", tbl.name)
        if tbl.name.lower() not in table_names:
            return (
                False,
                f"Table {tbl.name} not found. Please use the tables from the provided DDLs.",
            )

    # for col in tree.find_all(exp.Column):
    #     if isinstance(col, exp.Alias):
    #         column_names.add(
    #             col.alias
    #         )  # Add alias to column names as they might be referenced in the other clauses like order by
    #     print("Checking COLUMN NAME", col.name)
    #     if col.name.lower() not in column_names:
    #         return (
    #             False,
    #             f"Column {col.name} not found. Please use the columns from the provided DDLs.",
    #         )
    for col in tree.find_all(exp.Alias):
        column_names.add(
            col.alias.lower()
        )  # Add alias to column names as they might be referenced in the other clauses like order by
    
    for col in tree.find_all(exp.Column):
        print("Checking COLUMN NAME", col.name)
        if col.name.lower() not in column_names:
            return (
                False,
                f"Column {col.name} not found. Please use the columns from the provided DDLs.",
            )
        
    return (
        True,
        "The Generated SQL query is valid. Please move on to executing the query",
    )
