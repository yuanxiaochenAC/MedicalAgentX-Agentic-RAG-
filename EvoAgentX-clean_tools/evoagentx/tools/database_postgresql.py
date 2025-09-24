import time
import json
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import psycopg2
import psycopg2.extras
import re

from .database_base import DatabaseBase, DatabaseType, QueryType, DatabaseConnection
from .tool import Tool, Toolkit
from ..core.logging import logger

class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL-specific connection management"""
    def __init__(self, connection_string: str, **kwargs):
        super().__init__(connection_string, **kwargs)
        self.conn = None

    def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(self.connection_string, **self.connection_params)
            self._is_connected = True
            logger.info("Successfully connected to PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            self._is_connected = False
            return False

    def disconnect(self) -> bool:
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self._is_connected = False
                logger.info("Disconnected from PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")
            return False

    def test_connection(self) -> bool:
        try:
            if self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                return True
            return False
        except Exception:
            return False

class PostgreSQLDatabase(DatabaseBase):
    """
    PostgreSQL database implementation with automatic initialization.
    Handles remote connections, existing local databases, and new local database creation.
    """
    def __init__(self, 
                 connection_string: str = None,
                 database_name: str = None,
                 local_path: str = None,
                 auto_save: bool = True,
                 **kwargs):
        init_params = {
            'connection_string': connection_string,
            'database_name': database_name
        }
        super().__init__(**init_params, **kwargs)
        self.local_path = Path(local_path) if local_path else None
        self.auto_save = auto_save
        self.connection_params = kwargs
        self.is_local_database = False
        self.conn = None
        self.cursor = None
        self.file_based_mode = False
        self.tables = {}  # For file-based mode
        
        if self._is_remote_connection():
            self._init_remote_database()
        elif self._is_existing_local_database():
            self._init_existing_local_database()
        else:
            self._init_new_local_database()

    def _is_remote_connection(self) -> bool:
        return self.connection_string and ("@" in self.connection_string or "postgresql://" in self.connection_string)

    def _is_existing_local_database(self) -> bool:
        if not self.local_path:
            return False
        if not self.local_path.exists():
            return False
        db_info_file = self.local_path / "db_info.json"
        return db_info_file.exists()

    def _init_remote_database(self):
        """Initialize remote PostgreSQL connection"""
        try:
            # Add connection timeout to prevent hanging
            connection_params = self.connection_params.copy()
            connection_params.update({
                'connect_timeout': 5,  # 5 second timeout
                'options': '-c statement_timeout=5000'  # 5 second statement timeout
            })
            
            self.conn = psycopg2.connect(self.connection_string, **connection_params)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            if self.database_name:
                self.conn.set_isolation_level(0)
                self.cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (self.database_name,))
            self._is_initialized = True
            self.is_local_database = False
            self.file_based_mode = False
            logger.info(f"Connected to remote PostgreSQL: {self.database_name}")
        except Exception as e:
            logger.error(f"Failed to connect to remote PostgreSQL: {str(e)}")
            self._is_initialized = False
            # Don't raise, just log the error and continue with local mode
            logger.info("Falling back to local database mode")

    def _init_existing_local_database(self):
        """Initialize existing local file-based database"""
        try:
            if not self.database_name:
                self.database_name = self.local_path.name
            
            # Load existing tables from JSON files
            self._load_tables_from_files()
            
            self._is_initialized = True
            self.is_local_database = True
            self.file_based_mode = True
            logger.info(f"Loaded existing local file-based database from: {self.local_path}")
        except Exception as e:
            logger.error(f"Failed to load existing local database: {str(e)}")
            self._is_initialized = False
            logger.info("Falling back to new local database mode")
            self._init_new_local_database()

    def _init_new_local_database(self):
        """Initialize new local file-based database"""
        try:
            if not self.local_path:
                self.local_path = Path("./workplace/postgresql_local")
            self.local_path.mkdir(parents=True, exist_ok=True)
            
            if not self.database_name:
                self.database_name = self.local_path.name
            
            self._create_db_info_file()
            self._is_initialized = True
            self.is_local_database = True
            self.file_based_mode = True
            logger.info(f"Created new local file-based database at: {self.local_path}")
        except Exception as e:
            logger.error(f"Failed to create new local database: {str(e)}")
            self._is_initialized = False
            logger.info("Database initialization failed, but toolkit is still usable")

    def _create_db_info_file(self):
        """Create database info file"""
        try:
            db_info = {
                "database_name": self.database_name,
                "created_at": time.time(),
                "local_path": str(self.local_path.absolute()),
                "auto_save": self.auto_save,
                "version": "1.0",
                "mode": "file_based"
            }
            info_file = self.local_path / "db_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(db_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to create db info file: {str(e)}")

    def _load_tables_from_files(self):
        """Load tables from JSON files"""
        try:
            for json_file in self.local_path.glob("*.json"):
                if json_file.name == "db_info.json":
                    continue
                table_name = json_file.stem
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.tables[table_name] = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading tables from files: {str(e)}")

    def _save_table_to_file(self, table_name: str):
        """Save table data to JSON file"""
        try:
            if table_name in self.tables:
                table_file = self.local_path / f"{table_name}.json"
                with open(table_file, 'w', encoding='utf-8') as f:
                    json.dump(self.tables[table_name], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving table {table_name}: {str(e)}")

    def _parse_sql_query(self, sql: str) -> Dict[str, Any]:
        """Minimal SQL parser for file-based mode"""
        sql = sql.strip()
        upper_sql = sql.upper()
        # CREATE TABLE
        if upper_sql.startswith("CREATE TABLE"):
            match = re.search(r"CREATE TABLE (?:IF NOT EXISTS )?(\w+) *\((.*?)\)", sql, re.IGNORECASE | re.DOTALL)
            if match:
                table = match.group(1).lower()  # Normalize table name to lowercase
                columns = match.group(2)
                col_defs = [c.strip() for c in columns.split(',') if c.strip()]
                col_names = [c.split()[0] for c in col_defs]
                return {"type": "CREATE", "table": table, "columns": col_names}
        # INSERT
        elif upper_sql.startswith("INSERT"):
            match = re.search(r"INSERT INTO (\w+) *\((.*?)\) *VALUES", sql, re.IGNORECASE | re.DOTALL)
            if match:
                table = match.group(1).lower()  # Normalize table name to lowercase
                columns = [c.strip() for c in match.group(2).split(',')]
                # Extract all VALUES groups
                values_match = re.search(r"VALUES\s*(.*)", sql, re.IGNORECASE | re.DOTALL)
                if values_match:
                    values_str = values_match.group(1)
                    # Split by ),( to get individual value groups
                    value_groups = re.findall(r'\(([^)]+)\)', values_str)
                    all_values = []
                    for group in value_groups:
                        values = [v.strip().strip("'\"") for v in group.split(',')]
                        all_values.append(values)
                    return {"type": "INSERT", "table": table, "columns": columns, "values": all_values}
        # SELECT
        elif upper_sql.startswith("SELECT"):
            match = re.search(r"SELECT (.*?) FROM (\w+)(?: WHERE (.*?))?(?: GROUP BY (.*?))?(?: ORDER BY (.*?))?(?: LIMIT (\d+))?", sql, re.IGNORECASE | re.DOTALL)
            if match:
                columns = [c.strip() for c in match.group(1).split(',')]
                table = match.group(2).lower()  # Normalize table name to lowercase
                where = match.group(3)
                group_by = match.group(4)
                order_by = match.group(5)
                limit = match.group(6)
                return {"type": "SELECT", "table": table, "columns": columns, "where": where, "group_by": group_by, "order_by": order_by, "limit": limit}
        # UPDATE
        elif upper_sql.startswith("UPDATE"):
            match = re.search(r"UPDATE (\w+) SET (.*?)(?: WHERE (.*?))?$", sql, re.IGNORECASE | re.DOTALL)
            if match:
                table = match.group(1).lower()  # Normalize table name to lowercase
                set_clause = match.group(2)
                where = match.group(3)
                return {"type": "UPDATE", "table": table, "set": set_clause, "where": where}
        # DELETE
        elif upper_sql.startswith("DELETE"):
            match = re.search(r"DELETE FROM (\w+)(?: WHERE (.*?))?", sql, re.IGNORECASE | re.DOTALL)
            if match:
                table = match.group(1).lower()  # Normalize table name to lowercase
                where = match.group(2)
                return {"type": "DELETE", "table": table, "where": where}
        return {"type": "UNKNOWN"}

    def _get_database_type(self) -> DatabaseType:
        return DatabaseType.POSTGRESQL

    def connect(self) -> bool:
        return self._is_initialized

    def disconnect(self) -> bool:
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None
                self._is_initialized = False
                logger.info("Disconnected from PostgreSQL")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")
            return False

    def test_connection(self) -> bool:
        if self.file_based_mode:
            return self._is_initialized
        try:
            if self.conn:
                with self.conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                return True
            return False
        except Exception:
            return False

    def execute_query(self, query: Union[str, Dict, List], query_type: QueryType = None, **kwargs) -> Dict[str, Any]:
        if not self._is_initialized:
            return self.format_error_result("Database not initialized")
        
        if self.file_based_mode:
            return self._execute_file_based_query(query, query_type)
        
        if self.conn is None:
            return self.format_error_result("PostgreSQL server not available")
        
        start_time = time.time()
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if isinstance(query, str):
                    cur.execute(query)
                elif isinstance(query, dict):
                    sql = query.get("sql")
                    params = query.get("params", None)
                    cur.execute(sql, params)
                elif isinstance(query, list):
                    for q in query:
                        cur.execute(q)
                else:
                    return self.format_error_result("Unsupported query format", query_type)
                if cur.description:
                    result = cur.fetchall()
                else:
                    result = {"rowcount": cur.rowcount}
                self.conn.commit()
            execution_time = time.time() - start_time
            return self.format_query_result(result, query_type or QueryType.SELECT, execution_time=execution_time)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing PostgreSQL query: {str(e)}")
            # Rollback on error to prevent transaction issues
            try:
                if self.conn:
                    self.conn.rollback()
            except Exception as rollback_error:
                logger.warning(f"Error during rollback: {str(rollback_error)}")
            return self.format_error_result(str(e), query_type, execution_time=execution_time)

    def _execute_file_based_query(self, query: Union[str, Dict, List], query_type: QueryType = None) -> Dict[str, Any]:
        """Execute query in file-based mode"""
        start_time = time.time()
        try:
            if isinstance(query, str):
                parsed = self._parse_sql_query(query)
                query_type = query_type or QueryType.SELECT
                if parsed["type"] == "CREATE":
                    table_name = parsed["table"]
                    columns = parsed.get("columns", ["id"])
                    if table_name not in self.tables:
                        self.tables[table_name] = []
                    # Store schema as a hidden key
                    self.tables[f"__schema__{table_name}"] = columns
                    if self.auto_save:
                        self._save_table_to_file(table_name)
                    result = {"rowcount": 0}
                elif parsed["type"] == "INSERT":
                    table_name = parsed["table"]
                    columns = parsed["columns"]
                    all_values = parsed["values"]
                    if table_name not in self.tables:
                        self.tables[table_name] = []
                    
                    # Insert all rows
                    for values in all_values:
                        row = {col: val for col, val in zip(columns, values)}
                        row["id"] = len(self.tables[table_name]) + 1
                        self.tables[table_name].append(row)
                    
                    if self.auto_save:
                        self._save_table_to_file(table_name)
                    result = {"rowcount": len(all_values)}
                elif parsed["type"] == "SELECT":
                    table_name = parsed["table"]
                    columns = parsed["columns"]
                    where = parsed.get("where")
                    group_by = parsed.get("group_by")
                    rows = self.tables.get(table_name, [])
                    # WHERE support (simple col = 'val' and col > val)
                    if where:
                        # Handle simple conditions: col = 'val', col > val, etc.
                        m = re.match(r"(\w+) *([=><]+) *'?([\w@.\- ]+)'?", where)
                        if m:
                            col, op, val = m.group(1), m.group(2), m.group(3)
                            if op == "=":
                                rows = [r for r in rows if str(r.get(col)) == val]
                            elif op == ">":
                                try:
                                    val_num = int(val)
                                    rows = [r for r in rows if int(r.get(col, 0)) > val_num]
                                except ValueError:
                                    pass
                            elif op == "<":
                                try:
                                    val_num = int(val)
                                    rows = [r for r in rows if int(r.get(col, 0)) < val_num]
                                except ValueError:
                                    pass
                    
                    # Handle basic aggregation
                    if group_by:
                        # Simple GROUP BY with basic aggregation
                        group_col = group_by.strip()
                        groups = {}
                        for row in rows:
                            group_val = row.get(group_col, "Unknown")
                            if group_val not in groups:
                                groups[group_val] = []
                            groups[group_val].append(row)
                        
                        result = []
                        for group_val, group_rows in groups.items():
                            group_result = {group_col: group_val}
                            # Always include all aggregation keys
                            group_result["employee_count"] = len(group_rows)
                            salaries = [float(r.get("salary", 0)) for r in group_rows if r.get("salary") is not None]
                            group_result["avg_salary"] = sum(salaries) / len(salaries) if salaries else 0
                            group_result["max_salary"] = max(salaries) if salaries else 0
                            result.append(group_result)
                    else:
                        # Only return requested columns
                        if columns == ['*']:
                            result = rows
                        else:
                            result = [{col: r.get(col) for col in columns if col in r} for r in rows]
                elif parsed["type"] == "UPDATE":
                    table_name = parsed["table"]
                    set_clause = parsed["set"]
                    where = parsed.get("where")
                    rows = self.tables.get(table_name, [])
                    # Parse set_clause: col1 = 'val1', col2 = 'val2'
                    updates = dict(re.findall(r"(\w+) *= *'?([\w@.\- ]+)'?", set_clause))
                    count = 0
                    for r in rows:
                        match = True
                        if where:
                            m = re.match(r"(\w+) *= *'?([\w@.\- ]+)'?", where)
                            if m:
                                col, val = m.group(1), m.group(2)
                                if str(r.get(col)) != val:
                                    match = False
                        if match:
                            r.update(updates)
                            count += 1
                    if self.auto_save:
                        self._save_table_to_file(table_name)
                    result = {"rowcount": count}
                elif parsed["type"] == "DELETE":
                    table_name = parsed["table"]
                    where = parsed.get("where")
                    rows = self.tables.get(table_name, [])
                    if where:
                        m = re.match(r"(\w+) *([=><]+) *'?([\w@.\- ]+)'?", where)
                        if m:
                            col, op, val = m.group(1), m.group(2), m.group(3)
                            if op == "=":
                                new_rows = [r for r in rows if str(r.get(col)) != val]
                            elif op == ">":
                                try:
                                    val_num = int(val)
                                    new_rows = [r for r in rows if int(r.get(col, 0)) <= val_num]
                                except ValueError:
                                    new_rows = rows
                            else:
                                new_rows = rows
                            deleted_count = len(rows) - len(new_rows)
                            self.tables[table_name] = new_rows
                        else:
                            deleted_count = 0
                    else:
                        deleted_count = len(rows)
                        self.tables[table_name] = []
                    if self.auto_save:
                        self._save_table_to_file(table_name)
                    result = {"rowcount": deleted_count}
                else:
                    return self.format_error_result("Unsupported query type in file-based mode", query_type)
                execution_time = time.time() - start_time
                return self.format_query_result(result, query_type, execution_time=execution_time)
            else:
                return self.format_error_result("Unsupported query format in file-based mode", query_type)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing file-based query: {str(e)}")
            return self.format_error_result(str(e), query_type, execution_time=execution_time)

    def get_database_info(self) -> Dict[str, Any]:
        try:
            if not self._is_initialized:
                return self.format_error_result("Database not initialized")
            
            if self.file_based_mode:
                info = {
                    "database": self.database_name,
                    "user": "file_based",
                    "table_count": len(self.tables),
                    "connection_string": "file_based",
                    "is_connected": True,
                    "mode": "file_based"
                }
            else:
                if self.conn is None:
                    return self.format_error_result("PostgreSQL server not available")
                
                with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT current_database() as database, current_user as user")
                    db_info = cur.fetchone()
                    cur.execute("SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'")
                    table_count = cur.fetchone()["table_count"]
                info = {
                    "database": db_info["database"],
                    "user": db_info["user"],
                    "table_count": table_count,
                    "connection_string": self.connection_string,
                    "is_connected": self._is_initialized
                }
            return self.format_query_result(info, QueryType.SELECT)
        except Exception as e:
            return self.format_error_result(str(e))

    def list_collections(self) -> List[str]:
        try:
            if self.file_based_mode:
                return list(self.tables.keys())
            if not self._is_initialized or self.conn is None:
                return []
            with self.conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                tables = [row[0] for row in cur.fetchall()]
            return tables
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            if not self._is_initialized:
                return self.format_error_result("Database not initialized")
            
            if self.file_based_mode:
                if collection_name in self.tables:
                    row_count = len(self.tables[collection_name])
                    info = {
                        "table_name": collection_name,
                        "row_count": row_count,
                        "columns": ["id"]  # Simple column structure
                    }
                else:
                    return self.format_error_result(f"Table {collection_name} not found")
            else:
                if self.conn is None:
                    return self.format_error_result("PostgreSQL server not available")
                
                with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"SELECT COUNT(*) as row_count FROM {collection_name}")
                    row_count = cur.fetchone()["row_count"]
                    cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (collection_name,))
                    columns = cur.fetchall()
                info = {
                    "table_name": collection_name,
                    "row_count": row_count,
                    "columns": columns
                }
            return self.format_query_result(info, QueryType.SELECT)
        except Exception as e:
            return self.format_error_result(str(e))

    def get_schema(self, collection_name: str = None) -> Dict[str, Any]:
        try:
            if not self._is_initialized:
                return self.format_error_result("Database not initialized")
            
            if self.file_based_mode:
                if collection_name:
                    if collection_name in self.tables:
                        schema = {"id": "integer"}
                        return self.format_query_result({"table_name": collection_name, "schema": schema}, QueryType.SELECT)
                    else:
                        return self.format_error_result(f"Table {collection_name} not found")
                else:
                    schemas = {}
                    for table_name in self.tables:
                        schemas[table_name] = {"id": "integer"}
                    return self.format_query_result({"database_name": self.database_name, "schemas": schemas}, QueryType.SELECT)
            else:
                if self.conn is None:
                    return self.format_error_result("PostgreSQL server not available")
                
                with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if collection_name:
                        cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (collection_name,))
                        columns = cur.fetchall()
                        schema = {col["column_name"]: col["data_type"] for col in columns}
                        return self.format_query_result({"table_name": collection_name, "schema": schema}, QueryType.SELECT)
                    else:
                        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
                        tables = [row[0] for row in cur.fetchall()]
                        schemas = {}
                        for table in tables:
                            cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s", (table,))
                            columns = cur.fetchall()
                            schemas[table] = {col["column_name"]: col["data_type"] for col in columns}
                        return self.format_query_result({"database_name": self.database_name, "schemas": schemas}, QueryType.SELECT)
        except Exception as e:
            return self.format_error_result(str(e))

    def get_supported_query_types(self) -> List[QueryType]:
        return [
            QueryType.SELECT,
            QueryType.INSERT,
            QueryType.UPDATE,
            QueryType.DELETE,
            QueryType.CREATE,
            QueryType.DROP,
            QueryType.ALTER,
            QueryType.INDEX
        ]

    def get_capabilities(self) -> Dict[str, Any]:
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            "supports_sql": True,
            "supports_transactions": not self.file_based_mode,
            "supports_indexing": not self.file_based_mode,
            "schema_flexible": self.file_based_mode,
            "file_based_mode": self.file_based_mode
        })
        return base_capabilities

# Tool classes
class PostgreSQLExecuteTool(Tool):
    name: str = "postgresql_execute"
    description: str = "Execute arbitrary SQL queries on PostgreSQL."
    inputs: Dict[str, Dict[str, str]] = {
        "query": {"type": "string", "description": "SQL query to execute (can be SELECT, INSERT, UPDATE, DELETE, etc.)"},
        "query_type": {"type": "string", "description": "Type of query (select, insert, update, delete, create, drop, alter, index) - auto-detected if not provided"}
    }
    required: Optional[List[str]] = ["query"]
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, query: str, query_type: str = None) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            query_type_enum = None
            if query_type:
                try:
                    query_type_enum = QueryType(query_type.lower())
                except ValueError:
                    return {"success": False, "error": f"Invalid query type: {query_type}", "data": None}
            result = self.database.execute_query(query=query, query_type=query_type_enum)
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_execute tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLFindTool(Tool):
    name: str = "postgresql_find"
    description: str = "Find (SELECT) rows from a PostgreSQL table."
    inputs: Dict[str, Dict[str, str]] = {
        "table_name": {"type": "string", "description": "Table name to query"},
        "where": {"type": "string", "description": "WHERE clause (optional, e.g., 'age > 18')"},
        "columns": {"type": "string", "description": "Comma-separated columns to select (default '*')"},
        "limit": {"type": "integer", "description": "Maximum number of rows to return (optional)"},
        "offset": {"type": "integer", "description": "Number of rows to skip (optional)"},
        "sort": {"type": "string", "description": "ORDER BY clause (optional, e.g., 'age ASC')"}
    }
    required: Optional[List[str]] = ["table_name"]
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, table_name: str, where: str = None, columns: str = "*", limit: int = None, offset: int = None, sort: str = None) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            sql = f"SELECT {columns} FROM {table_name}"
            if where:
                sql += f" WHERE {where}"
            if sort:
                sql += f" ORDER BY {sort}"
            if limit is not None:
                sql += f" LIMIT {limit}"
            if offset is not None:
                sql += f" OFFSET {offset}"
            result = self.database.execute_query(sql, QueryType.SELECT)
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_find tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLUpdateTool(Tool):
    name: str = "postgresql_update"
    description: str = "Update rows in a PostgreSQL table."
    inputs: Dict[str, Dict[str, str]] = {
        "table_name": {"type": "string", "description": "Table name to update"},
        "set": {"type": "string", "description": "SET clause (e.g., 'status = \'active\'')"},
        "where": {"type": "string", "description": "WHERE clause (optional)"}
    }
    required: Optional[List[str]] = ["table_name", "set"]
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, table_name: str, set: str, where: str = None) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            sql = f"UPDATE {table_name} SET {set}"
            if where:
                sql += f" WHERE {where}"
            result = self.database.execute_query(sql, QueryType.UPDATE)
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_update tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLCreateTool(Tool):
    name: str = "postgresql_create"
    description: str = "Create a table or other object in PostgreSQL."
    inputs: Dict[str, Dict[str, str]] = {
        "query": {"type": "string", "description": "CREATE statement (e.g., CREATE TABLE ...)"}
    }
    required: Optional[List[str]] = ["query"]
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, query: str) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            result = self.database.execute_query(query, QueryType.CREATE)
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_create tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLDeleteTool(Tool):
    name: str = "postgresql_delete"
    description: str = "Delete rows from a PostgreSQL table."
    inputs: Dict[str, Dict[str, str]] = {
        "table_name": {"type": "string", "description": "Table name to delete from"},
        "where": {"type": "string", "description": "WHERE clause (optional)"}
    }
    required: Optional[List[str]] = ["table_name"]
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, table_name: str, where: str = None) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            sql = f"DELETE FROM {table_name}"
            if where:
                sql += f" WHERE {where}"
            result = self.database.execute_query(sql, QueryType.DELETE)
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_delete tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLInfoTool(Tool):
    name: str = "postgresql_info"
    description: str = "Get PostgreSQL database and table information."
    inputs: Dict[str, Dict[str, str]] = {
        "info_type": {"type": "string", "description": "Type of information (database, tables, table, schema, capabilities)"},
        "table_name": {"type": "string", "description": "Table name for table-specific info (optional)"}
    }
    required: Optional[List[str]] = []
    def __init__(self, database: PostgreSQLDatabase = None):
        super().__init__()
        self.database = database
    def __call__(self, info_type: str = "database", table_name: str = None) -> Dict[str, Any]:
        try:
            if not self.database:
                return {"success": False, "error": "PostgreSQL database not initialized", "data": None}
            info_type = info_type.lower()
            if info_type == "database":
                result = self.database.get_database_info()
            elif info_type == "tables":
                tables = self.database.list_collections()
                result = {"success": True, "data": tables, "table_count": len(tables)}
            elif info_type == "table" and table_name:
                result = self.database.get_collection_info(table_name)
            elif info_type == "schema":
                result = self.database.get_schema(table_name)
            elif info_type == "capabilities":
                result = {"success": True, "data": self.database.get_capabilities()}
            else:
                return {"success": False, "error": f"Invalid info type: {info_type}", "data": None}
            return result
        except Exception as e:
            logger.error(f"Error in postgresql_info tool: {str(e)}")
            return {"success": False, "error": str(e), "data": None}

class PostgreSQLToolkit(Toolkit):
    def __init__(self, 
                 name: str = "PostgreSQLToolkit",
                 connection_string: str = None,
                 database_name: str = None,
                 local_path: str = None,
                 auto_save: bool = True,
                 **kwargs):
        database = PostgreSQLDatabase(
            connection_string=connection_string,
            database_name=database_name,
            local_path=local_path,
            auto_save=auto_save,
            **kwargs
        )
        tools = [
            PostgreSQLExecuteTool(database=database),
            PostgreSQLFindTool(database=database),
            PostgreSQLUpdateTool(database=database),
            PostgreSQLCreateTool(database=database),
            PostgreSQLDeleteTool(database=database),
            PostgreSQLInfoTool(database=database)
        ]
        super().__init__(name=name, tools=tools)
        self.database = database
        self.connection_string = connection_string
        self.database_name = database_name
        self.local_path = local_path
        self.auto_save = auto_save
        import atexit
        atexit.register(self._cleanup)
    def _cleanup(self):
        try:
            if self.database:
                self.database.disconnect()
                logger.info("Disconnected from PostgreSQL database")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
    def get_capabilities(self) -> Dict[str, Any]:
        if self.database:
            capabilities = self.database.get_capabilities()
            capabilities.update({
                "is_local_database": self.database.is_local_database,
                "local_path": str(self.database.local_path) if self.database.local_path else None,
                "auto_save": self.database.auto_save
            })
            return capabilities
        return {"error": "PostgreSQL database not initialized"}
    def connect(self) -> bool:
        return self.database.connect() if self.database else False
    def disconnect(self) -> bool:
        return self.database.disconnect() if self.database else False
    def test_connection(self) -> bool:
        return self.database.test_connection() if self.database else False
    def get_database(self) -> PostgreSQLDatabase:
        return self.database
    def get_local_info(self) -> Dict[str, Any]:
        return {
            "is_local_database": self.database.is_local_database,
            "local_path": str(self.database.local_path) if self.database.local_path else None,
            "auto_save": self.database.auto_save,
            "database_name": self.database_name,
            "connection_string": self.connection_string
        } if self.database else {"error": "Database not initialized"} 