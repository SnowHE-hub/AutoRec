from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class DimUser(Base):
    __tablename__ = "dim_user"

    user_id           = Column(String(10),  primary_key=True)
    age               = Column(SmallInteger, nullable=False)
    gender            = Column(String(1),   nullable=False)
    income_bracket    = Column(String(4),   nullable=False)
    city              = Column(String(100), nullable=False)
    registration_date = Column(Date,        nullable=False)

    interactions = relationship(
        "FactInteraction", back_populates="user", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<DimUser {self.user_id} age={self.age} income={self.income_bracket}>"


class DimCar(Base):
    __tablename__ = "dim_car"

    car_id          = Column(String(10),  primary_key=True)
    make            = Column(String(50),  nullable=False)
    model           = Column(String(100), nullable=False)
    year            = Column(SmallInteger, nullable=False)
    body_type       = Column(String(20),  nullable=False)
    fuel_type       = Column(String(20),  nullable=False)
    transmission    = Column(String(20),  nullable=False)
    price           = Column(Integer,     nullable=False)
    price_tier      = Column(String(10),  nullable=False)
    mileage         = Column(Integer,     nullable=False)
    age_at_listing  = Column(SmallInteger, nullable=False)

    interactions = relationship(
        "FactInteraction", back_populates="car", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<DimCar {self.car_id} {self.make} {self.model} {self.year}>"


class DimDate(Base):
    __tablename__ = "dim_date"

    date_id     = Column(Integer,     primary_key=True)  # YYYYMMDD
    full_date   = Column(Date,        nullable=False, unique=True)
    year        = Column(SmallInteger, nullable=False)
    quarter     = Column(SmallInteger, nullable=False)
    month       = Column(SmallInteger, nullable=False)
    day_of_week = Column(SmallInteger, nullable=False)  # 0=Monday (Python weekday)
    is_weekend  = Column(Boolean,      nullable=False)

    interactions = relationship(
        "FactInteraction", back_populates="date", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<DimDate {self.date_id} {self.full_date}>"


class FactInteraction(Base):
    __tablename__ = "fact_interactions"

    interaction_id        = Column(String(10),  primary_key=True)
    user_id               = Column(String(10),  ForeignKey("dim_user.user_id"), nullable=False)
    car_id                = Column(String(10),  ForeignKey("dim_car.car_id"),   nullable=False)
    date_id               = Column(Integer,     ForeignKey("dim_date.date_id"), nullable=False)
    interaction_type      = Column(String(15),  nullable=False)
    interaction_timestamp = Column(DateTime,    nullable=False)

    user = relationship("DimUser", back_populates="interactions")
    car  = relationship("DimCar",  back_populates="interactions")
    date = relationship("DimDate", back_populates="interactions")

    def __repr__(self) -> str:
        return (
            f"<FactInteraction {self.interaction_id} "
            f"u={self.user_id} c={self.car_id} type={self.interaction_type}>"
        )
