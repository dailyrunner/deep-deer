"""Natural Language Query processor for SQL generation"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

from app.core.database import get_all_tables, get_table_info, engine
from app.services.ollama_service import ollama_service
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class NLQResult:
    """Result of NLQ processing"""
    sql: str
    success: bool
    data: Optional[List[Dict]] = None
    error: Optional[str] = None
    tables_used: Optional[List[str]] = None


class NLQProcessor:
    """Process natural language queries to SQL"""

    def __init__(self):
        self.is_ready = False
        self.schema_cache = {}
        self.examples = self._load_examples()

    async def initialize(self):
        """Initialize NLQ processor"""
        try:
            # Load database schema
            await self._refresh_schema()
            self.is_ready = True
            logger.info("NLQ processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLQ processor: {e}")
            self.is_ready = False

    async def _refresh_schema(self):
        """Refresh database schema information"""
        try:
            tables = await get_all_tables()
            self.schema_cache = {}

            for table in tables:
                table_info = await get_table_info(table)
                self.schema_cache[table] = table_info

            logger.info(f"Schema refreshed: {len(self.schema_cache)} tables")
        except Exception as e:
            logger.error(f"Failed to refresh schema: {e}")

    def _load_examples(self) -> List[Dict[str, str]]:
        """Load few-shot examples for SQL generation"""
        return [
            {
                "question": "고객 목록을 보여줘",
                "sql": "SELECT * FROM customers;"
            },
            {
                "question": "Show all products",
                "sql": "SELECT * FROM products;"
            },
            {
                "question": "월별 매출 합계를 계산해줘",
                "sql": "SELECT DATE_FORMAT(order_date, '%Y-%m') as month, SUM(total_amount) as total_sales FROM orders GROUP BY month ORDER BY month;"
            },
            {
                "question": "상위 10명의 고객을 매출액 기준으로 보여줘",
                "sql": "SELECT c.name, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 10;"
            },
            {
                "question": "재고가 10개 미만인 제품을 찾아줘",
                "sql": "SELECT * FROM products WHERE stock_quantity < 10;"
            }
        ]

    def _generate_schema_info(self) -> str:
        """Generate schema information for LLM"""
        schema_lines = []

        for table_name, table_info in self.schema_cache.items():
            schema_lines.append(f"Table: {table_name}")
            schema_lines.append("Columns:")

            for col in table_info['columns']:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                pk = " PRIMARY KEY" if col['primary_key'] else ""
                schema_lines.append(f"  - {col['name']} ({col['type']}) {nullable}{pk}")

            schema_lines.append("")

        return "\n".join(schema_lines)

    def _extract_tables_from_sql(self, sql: str) -> List[str]:
        """Extract table names from SQL query"""
        tables = []
        sql_upper = sql.upper()

        # Extract from FROM clause
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        from_matches = re.findall(from_pattern, sql_upper)
        tables.extend(from_matches)

        # Extract from JOIN clauses
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        join_matches = re.findall(join_pattern, sql_upper)
        tables.extend(join_matches)

        # Convert to lowercase and remove duplicates
        tables = list(set(t.lower() for t in tables))
        return tables

    def _validate_sql(self, sql: str) -> bool:
        """Basic SQL validation"""
        sql_upper = sql.upper().strip()

        # Check for dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                logger.warning(f"Dangerous SQL keyword detected: {keyword}")
                return False

        # Check if it's a SELECT statement
        if not sql_upper.startswith('SELECT'):
            logger.warning("Only SELECT statements are allowed")
            return False

        return True

    async def process_query(self, question: str) -> NLQResult:
        """Process natural language query to SQL and execute"""
        if not self.is_ready:
            return NLQResult(
                sql="",
                success=False,
                error="NLQ processor is not ready"
            )

        try:
            # Generate SQL
            schema_info = self._generate_schema_info()
            sql = await ollama_service.generate_sql(
                question=question,
                schema_info=schema_info,
                examples=self.examples
            )

            # Validate SQL
            if not self._validate_sql(sql):
                return NLQResult(
                    sql=sql,
                    success=False,
                    error="Generated SQL failed validation"
                )

            # Extract tables used
            tables_used = self._extract_tables_from_sql(sql)

            # Execute SQL
            async with engine.connect() as conn:
                result = await conn.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()

                # Convert to list of dicts
                data = [dict(zip(columns, row)) for row in rows]

            return NLQResult(
                sql=sql,
                success=True,
                data=data,
                tables_used=tables_used
            )

        except Exception as e:
            logger.error(f"Error processing NLQ: {e}")
            return NLQResult(
                sql=sql if 'sql' in locals() else "",
                success=False,
                error=str(e)
            )

    async def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the database schema"""
        await self._refresh_schema()

        summary = {
            "tables_count": len(self.schema_cache),
            "tables": []
        }

        for table_name, table_info in self.schema_cache.items():
            summary["tables"].append({
                "name": table_name,
                "columns_count": len(table_info['columns']),
                "columns": [col['name'] for col in table_info['columns']]
            })

        return summary


# Global instance
nlq_processor = NLQProcessor()