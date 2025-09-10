"""Add sub_type field to trains table

Revision ID: add_train_sub_type
Revises: 
Create Date: 2025-09-07 23:15:32.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = 'add_train_sub_type'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if sub_type column already exists
    connection = op.get_bind()
    result = connection.execute(
    text("SELECT column_name FROM information_schema.columns WHERE table_name='trains' AND column_name='sub_type'")
    )
    if not result.fetchone():
        # Add sub_type column to trains table
        op.add_column('trains', sa.Column('sub_type', sa.String(100), nullable=True))
    
    # Update the enum values for train type - use a safer approach
    try:
        op.execute("ALTER TYPE traintype ADD VALUE IF NOT EXISTS 'Superfast'")
        op.execute("ALTER TYPE traintype ADD VALUE IF NOT EXISTS 'Shatabdi'")
        op.execute("ALTER TYPE traintype ADD VALUE IF NOT EXISTS 'Rajdhani'")
        op.execute("ALTER TYPE traintype ADD VALUE IF NOT EXISTS 'Duronto'")
        op.execute("ALTER TYPE traintype ADD VALUE IF NOT EXISTS 'GaribRath'")
    except Exception:
        # If enum doesn't exist or has issues, we'll handle it gracefully
        pass


def downgrade() -> None:
    # Check if sub_type column exists before dropping
    connection = op.get_bind()
    result = connection.execute(
        text("SELECT column_name FROM information_schema.columns WHERE table_name='trains' AND column_name='sub_type'")
    )
    if result.fetchone():
        op.drop_column('trains', 'sub_type')
