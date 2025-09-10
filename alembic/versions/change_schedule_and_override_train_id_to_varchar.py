"""Change schedules.train_id and overrides.train_id to VARCHAR and backfill values

Revision ID: change_train_id_to_varchar
Revises: add_train_sub_type
Create Date: 2025-09-09 02:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = 'change_train_id_to_varchar'
down_revision = 'add_train_sub_type'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1) Add temporary column with desired type
    op.add_column('schedules', sa.Column('train_id_str', sa.String(50), nullable=True))
    op.add_column('overrides', sa.Column('train_id_str', sa.String(50), nullable=True))

    # 2) Backfill from integer FK (numeric id) to human-readable trains.train_id
    # schedules: join schedules.train_id (old int) to trains.id and copy trains.train_id
    conn.execute(text('''
        UPDATE schedules s
        SET train_id_str = t.train_id
        FROM trains t
        WHERE CAST(s.train_id AS INTEGER) = t.id
    '''))

    # overrides
    try:
        conn.execute(text('''
            UPDATE overrides o
            SET train_id_str = t.train_id
            FROM trains t
            WHERE CAST(o.train_id AS INTEGER) = t.id
        '''))
    except Exception:
        # overrides table may be empty or not used yet; continue gracefully
        pass

    # 3) Drop old columns and rename new
    with op.batch_alter_table('schedules') as batch_op:
        batch_op.drop_column('train_id')
        batch_op.alter_column('train_id_str', new_column_name='train_id', existing_type=sa.String(50), nullable=False)

    try:
        with op.batch_alter_table('overrides') as batch_op:
            batch_op.drop_column('train_id')
            batch_op.alter_column('train_id_str', new_column_name='train_id', existing_type=sa.String(50), nullable=False)
    except Exception:
        pass


def downgrade() -> None:
    # Recreate integer columns (will be nullable to avoid data loss) and attempt reverse map
    op.add_column('schedules', sa.Column('train_id_int', sa.Integer(), nullable=True))
    op.add_column('overrides', sa.Column('train_id_int', sa.Integer(), nullable=True))

    conn = op.get_bind()
    # Reverse map using trains table
    conn.execute(text('''
        UPDATE schedules s
        SET train_id_int = t.id
        FROM trains t
        WHERE s.train_id = t.train_id
    '''))
    try:
        conn.execute(text('''
            UPDATE overrides o
            SET train_id_int = t.id
            FROM trains t
            WHERE o.train_id = t.train_id
        '''))
    except Exception:
        pass

    with op.batch_alter_table('schedules') as batch_op:
        batch_op.drop_column('train_id')
        batch_op.alter_column('train_id_int', new_column_name='train_id', existing_type=sa.Integer(), nullable=True)

    try:
        with op.batch_alter_table('overrides') as batch_op:
            batch_op.drop_column('train_id')
            batch_op.alter_column('train_id_int', new_column_name='train_id', existing_type=sa.Integer(), nullable=True)
    except Exception:
        pass