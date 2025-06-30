import base64
import os
from io import BytesIO

from sqlalchemy import Column, create_engine, text, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from sqlalchemy.types import Boolean, DateTime, Integer, Text

import json


Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True)
    user_question = Column(Text, nullable=False)
    query = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    figure = Column(LargeBinary, nullable=True)
    figexp = Column(Text, nullable=False)
    # is_user = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False, default=func.now())


engine = create_engine(os.environ["BACKEND_DB_URL"])
Base.metadata.create_all(engine)
Session = sessionmaker(engine)


def encode_image(figure):
    if not figure:
        return None

    buf = BytesIO()
    figure.save(buf, format="PNG")
    buf.seek(0)
    base64_encoded_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return base64_encoded_image

def encode_plotly(figure):
    if not figure:
        return None
    # Serialize to JSON (or you can store it as a binary object)
    graph_json = figure.to_json()

    # Convert to binary for storage (use json.dumps() for JSON storage)
    graph_binary = json.dumps(graph_json).encode('utf-8')

    return graph_binary

def add_to_chat_history(messages, figure=None):
    with Session() as sess:
        for m in messages:
            sess.add(
                # ChatHistory(content=m[0], is_user=m[1], user_question = m[2], figure=encode_plotly(figure))
                ChatHistory(query=m[0], explanation = m[1], user_question = m[2], figexp =m[3], figure=encode_plotly(figure))
            )
        sess.commit()


def get_chat_history():
    records = []
    with Session() as sess:
        for rcd in (
            sess.query(ChatHistory).order_by(ChatHistory.created_at.desc()).limit(15) #.filter(ChatHistory.created_at >= start_time)
        ):
            records.append(rcd)
    records.reverse()
    return records
