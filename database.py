from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Use environment variable or default to local PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/loan_risk")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    income = Column(Float)
    loan_amount = Column(Float)
    credit_score = Column(Integer)
    probability = Column(Float)
    risk_category = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
