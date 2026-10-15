
from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    CheckConstraint,
    func,
)
from sqlalchemy.orm import relationship

from database import Base


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)

    invoice_number = Column(String(100), nullable=True)
    vendor = Column(String(200), nullable=False)
    invoice_date = Column(Date, nullable=True)
    invoice_type = Column(
        String(20), nullable=False, default="unknown"
    )
    currency = Column(String(3), nullable=True)

    subtotal = Column(Numeric(12, 2), nullable=False)
    tax = Column(Numeric(12, 2), nullable=False)
    total = Column(Numeric(12, 2), nullable=False)

    status = Column(
        String(30), nullable=False, default="confirmed"
    )
    created_at = Column(
        DateTime, nullable=False, server_default=func.now()
    )

    items = relationship(
        "InvoiceItem",
        back_populates="invoice",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint("subtotal >= 0"),
        CheckConstraint("tax >= 0"),
        CheckConstraint("total >= 0"),
    )


class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    id = Column(Integer, primary_key=True, index=True)

    invoice_id = Column(
        Integer,
        ForeignKey("invoices.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name = Column(String(200), nullable=False)
    quantity = Column(Numeric(12, 3), nullable=True)
    unit_price = Column(Numeric(12, 2), nullable=True)
    amount = Column(Numeric(12, 2), nullable=True)

    invoice = relationship(
        "Invoice",
        back_populates="items",
    )

    __table_args__ = (
        CheckConstraint("quantity IS NULL OR quantity > 0"),
        CheckConstraint("unit_price IS NULL OR unit_price >= 0"),
        CheckConstraint("amount IS NULL OR amount >= 0"),
    )
