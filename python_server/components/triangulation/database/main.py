from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from python_server.settings.settings import settings
from python_server.components.triangulation.database.base import Base
from python_server.components.triangulation.database.uv_xyz import UV_XYZ 


dbName = settings().database.name
DATABASE_URL = f'sqlite:///{dbName}.db'
print("DATABASE_URL",DATABASE_URL)


engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
      # Ensure the models are imported before creating tables
    Base.metadata.create_all(bind=engine)
    print("tables",Base.metadata.tables.keys())
    print("init_db DONE")


def empty_table():
    print("empty_table")
    session = SessionLocal()
    try:

        session.execute(text(f"DELETE FROM {UV_XYZ.__tablename__}"))
        session.commit()
        print(f"Table {UV_XYZ.__tablename__} has been emptied.")

    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()
