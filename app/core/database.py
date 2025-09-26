"""Database connection and session management"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Create base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_sqlite_inmemory():
    """Initialize SQLite with test data"""
    async with engine.begin() as conn:
        # 1. 과목 테이블
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id TEXT PRIMARY KEY,
            subject_name TEXT,
            subject_intro TEXT
        )"""))

        # 2. 교수진 테이블
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS professors (
            professor_id TEXT PRIMARY KEY,
            professor_name TEXT
        )"""))

        # 3. 학생 테이블
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            student_name TEXT,
            student_number TEXT,
            major TEXT
        )"""))

        # 4. 커리큘럼 테이블
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS curriculum (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            professor_id TEXT,
            subject_id TEXT,
            date TEXT
        )"""))

        # 5. 피드백 테이블
        await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            subject_id TEXT,
            feedback_content TEXT
        )"""))

        # 과목 데이터 삽입
        res = await conn.execute(text("SELECT COUNT(*) FROM subjects"))
        count = res.scalar() or 0
        if count == 0:
            await conn.execute(text("""
                INSERT INTO subjects (subject_id, subject_name, subject_intro) VALUES
                ('CS101', 'Cloud 기반 컨테이너 기술 이해', 'Docker와 Kubernetes를 활용한 클라우드 컨테이너 기술의 이론과 실습을 다룹니다'),
                ('CS201', 'Prompt Engineering', 'AI 모델과의 효과적인 상호작용을 위한 프롬프트 설계 및 최적화 기법을 학습합니다'),
                ('CS301', 'DevOps 이해', 'CI/CD 파이프라인 구축, 자동화, 모니터링 등 DevOps 문화와 실무를 학습합니다'),
                ('MATH101', 'DBMS 및 SQL 활용', '관계형 데이터베이스 설계와 SQL 쿼리 작성 실무 능력을 배양합니다'),
                ('ENG101', 'RestAPI 구현', 'RESTful API 설계 원칙과 구현, API 문서화 및 테스팅을 학습합니다')
            """))

        # 교수진 데이터 삽입
        res2 = await conn.execute(text("SELECT COUNT(*) FROM professors"))
        count2 = res2.scalar() or 0
        if count2 == 0:
            await conn.execute(text("""
                INSERT INTO professors (professor_id, professor_name) VALUES
                ('P001', '이용우'),
                ('P002', '박병선'),
                ('P003', '배기주'),
                ('P004', '백정열'),
                ('P005', '이상민')
            """))

        # 학생 데이터 삽입
        res3 = await conn.execute(text("SELECT COUNT(*) FROM students"))
        count3 = res3.scalar() or 0
        if count3 == 0:
            await conn.execute(text("""
                INSERT INTO students (student_id, student_name, student_number, major) VALUES
                ('S001', '김민수', '2021001', '컴퓨터과학과'),
                ('S002', '이서연', '2022002', '컴퓨터과학과'),
                ('S003', '박준혁', '2023003', '수학과'),
                ('S004', '정하은', '2021004', '컴퓨터과학과'),
                ('S005', '최동현', '2022005', '경영학과')
            """))

        # 커리큘럼 데이터 삽입
        res4 = await conn.execute(text("SELECT COUNT(*) FROM curriculum"))
        count4 = res4.scalar() or 0
        if count4 == 0:
            await conn.execute(text("""
                INSERT INTO curriculum (professor_id, subject_id, date) VALUES
                ('P001', 'CS101', '2024-03-04'),
                ('P002', 'CS201', '2024-03-05'),
                ('P003', 'CS301', '2024-09-02'),
                ('P004', 'MATH101', '2024-03-06'),
                ('P005', 'ENG101', '2024-03-07')
            """))

        # 피드백 데이터 삽입
        res5 = await conn.execute(text("SELECT COUNT(*) FROM feedback"))
        count5 = res5.scalar() or 0
        if count5 == 0:
            await conn.execute(text("""
                INSERT INTO feedback (student_id, subject_id, feedback_content) VALUES
                ('S001', 'CS101', 'Docker 컨테이너 개념이 처음엔 어려웠지만, 실습을 통해 점차 이해할 수 있었습니다. Kubernetes는 더 많은 연습이 필요할 것 같아요.'),
                ('S002', 'CS201', 'GPT 모델과 대화하는 방법을 체계적으로 배울 수 있었습니다. 하지만 영어 프롬프트 작성이 아직 어렵네요.'),
                ('S003', 'MATH101', '테이블 조인과 서브쿼리 개념이 복잡해서 이해하는데 시간이 걸렸습니다. 더 많은 예제가 있으면 좋겠어요.'),
                ('S004', 'CS301', 'CI/CD 파이프라인 구축이 생각보다 복잡했습니다. Jenkins와 GitHub Actions 중 어떤 것을 선택해야 할지 고민됩니다.'),
                ('S001', 'CS201', 'Few-shot learning과 Chain-of-Thought 기법이 흥미로웠지만, 실제 적용하는 것이 어려웠습니다.')
            """))

        logger.info("Test data initialized successfully")


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")

    # Initialize test data
    await init_sqlite_inmemory()


async def test_connection():
    """Test database connection"""
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.scalar()
            logger.info("Database connection successful")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


async def get_table_info(table_name: str) -> dict:
    """Get table schema information"""
    async with engine.connect() as conn:
        # Get column information
        result = await conn.execute(
            text(f"PRAGMA table_info({table_name})")
        )
        columns = result.fetchall()

        return {
            "table_name": table_name,
            "columns": [
                {
                    "name": col[1],
                    "type": col[2],
                    "nullable": not col[3],
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]
        }


async def get_all_tables() -> list:
    """Get all table names in the database"""
    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = result.fetchall()
        return [table[0] for table in tables if not table[0].startswith("sqlite_")]