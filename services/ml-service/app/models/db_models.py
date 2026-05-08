import uuid
from datetime import datetime, date, time
from typing import List, Optional, Any, Dict
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Date, Time, 
    DateTime, ForeignKey, JSON, select, update, delete
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from pgvector.sqlalchemy import Vector
import os

Base = declarative_base()

# --- Database Setup ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/mira")

engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    echo=False
)

AsyncSessionFactory = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# --- ORM Models ---

class UserProfile(Base):
    __tablename__ = 'user_profiles'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    attachment_style: Mapped[Optional[str]] = mapped_column(String)
    big_five_scores: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    ocean_confidence: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    maslow_baseline: Mapped[Optional[int]] = mapped_column(Integer)
    onboarding_complete: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    analysis_modes: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    preferred_theme: Mapped[Optional[str]] = mapped_column(String, default='midnight')
    streak_count: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    longest_streak: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    last_checkin_date: Mapped[Optional[date]] = mapped_column(Date)
    total_sessions: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    weekly_email: Mapped[Optional[bool]] = mapped_column(Boolean, default=True)
    daily_reminder: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    reminder_time: Mapped[Optional[time]] = mapped_column(Time)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = 'sessions'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('user_profiles.id', ondelete='CASCADE'))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    modalities_used: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    dominant_emotion: Mapped[str] = mapped_column(String, nullable=False)
    emotion_vector: Mapped[Optional[Any]] = mapped_column(Vector(64))
    valence: Mapped[Optional[float]] = mapped_column(Float)
    arousal: Mapped[Optional[float]] = mapped_column(Float)
    dominance: Mapped[Optional[float]] = mapped_column(Float)
    intensity_level: Mapped[Optional[int]] = mapped_column(Integer)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer)
    drift_alert_level: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    crisis_level: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    conflict_detected: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    masked_emotion: Mapped[Optional[str]] = mapped_column(String)
    maslow_level: Mapped[Optional[int]] = mapped_column(Integer)
    cognitive_distortions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    theory_applied: Mapped[Optional[str]] = mapped_column(String)
    prescription_given: Mapped[Optional[str]] = mapped_column(String)
    notes: Mapped[Optional[str]] = mapped_column(String)

    user = relationship("UserProfile", back_populates="sessions")


class EmotionScore(Base):
    __tablename__ = 'emotion_scores'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('sessions.id', ondelete='CASCADE'))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('user_profiles.id', ondelete='CASCADE'))
    emotion_name: Mapped[str] = mapped_column(String, nullable=False)
    plutchik_category: Mapped[Optional[str]] = mapped_column(String)
    intensity_level: Mapped[Optional[int]] = mapped_column(Integer)
    score: Mapped[Optional[float]] = mapped_column(Float)
    source: Mapped[Optional[str]] = mapped_column(String)
    captured_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class PrescriptionGiven(Base):
    __tablename__ = 'prescriptions_given'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('user_profiles.id', ondelete='CASCADE'))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('sessions.id', ondelete='CASCADE'))
    prescription_name: Mapped[str] = mapped_column(String, nullable=False)
    theory_basis: Mapped[Optional[str]] = mapped_column(String)
    target_emotions: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    instructions: Mapped[str] = mapped_column(String, nullable=False)
    difficulty_level: Mapped[Optional[int]] = mapped_column(Integer)
    completed: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    user_rating: Mapped[Optional[int]] = mapped_column(Integer)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    given_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class CrisisEvent(Base):
    __tablename__ = 'crisis_events'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('user_profiles.id', ondelete='CASCADE'))
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey('sessions.id'))
    crisis_level: Mapped[Optional[int]] = mapped_column(Integer)
    detection_layers: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    triggered_by: Mapped[Optional[str]] = mapped_column(String)
    response_given: Mapped[Optional[str]] = mapped_column(String)
    resources_shown: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    resolved: Mapped[Optional[bool]] = mapped_column(Boolean)
    followup_sent: Mapped[Optional[bool]] = mapped_column(Boolean, default=False)
    occurred_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class ConversationSummary(Base):
    __tablename__ = 'conversation_summaries'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('user_profiles.id', ondelete='CASCADE'))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('sessions.id', ondelete='CASCADE'))
    summary: Mapped[str] = mapped_column(String, nullable=False)
    key_topics: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))
    dominant_emotion: Mapped[Optional[str]] = mapped_column(String)
    maslow_level: Mapped[Optional[int]] = mapped_column(Integer)
    chroma_doc_id: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


# --- Repositories ---

class UserProfileRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: uuid.UUID) -> Optional[UserProfile]:
        result = await self.session.execute(select(UserProfile).where(UserProfile.id == user_id))
        return result.scalar_one_or_none()

    async def create(self, user: UserProfile) -> UserProfile:
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update(self, user_id: uuid.UUID, **kwargs) -> Optional[UserProfile]:
        await self.session.execute(
            update(UserProfile).where(UserProfile.id == user_id).values(**kwargs)
        )
        await self.session.commit()
        return await self.get_by_id(user_id)


class SessionRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, session_id: uuid.UUID) -> Optional[Session]:
        result = await self.session.execute(select(Session).where(Session.id == session_id))
        return result.scalar_one_or_none()

    async def create(self, session_data: Session) -> Session:
        self.session.add(session_data)
        await self.session.commit()
        await self.session.refresh(session_data)
        return session_data

    async def get_sessions_by_user(self, user_id: uuid.UUID, limit: int = 10) -> List[Session]:
        result = await self.session.execute(
            select(Session).where(Session.user_id == user_id).order_by(Session.started_at.desc()).limit(limit)
        )
        return list(result.scalars().all())


class EmotionScoreRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, score: EmotionScore) -> EmotionScore:
        self.session.add(score)
        await self.session.commit()
        await self.session.refresh(score)
        return score

    async def get_scores_for_session(self, session_id: uuid.UUID) -> List[EmotionScore]:
        result = await self.session.execute(
            select(EmotionScore).where(EmotionScore.session_id == session_id)
        )
        return list(result.scalars().all())


class PrescriptionRepo:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, prescription: PrescriptionGiven) -> PrescriptionGiven:
        self.session.add(prescription)
        await self.session.commit()
        await self.session.refresh(prescription)
        return prescription

    async def get_pending_prescriptions(self, user_id: uuid.UUID) -> List[PrescriptionGiven]:
        result = await self.session.execute(
            select(PrescriptionGiven).where(
                PrescriptionGiven.user_id == user_id, 
                PrescriptionGiven.completed == False
            )
        )
        return list(result.scalars().all())

    async def mark_completed(self, prescription_id: uuid.UUID, rating: int) -> Optional[PrescriptionGiven]:
        await self.session.execute(
            update(PrescriptionGiven)
            .where(PrescriptionGiven.id == prescription_id)
            .values(completed=True, user_rating=rating, completed_at=datetime.utcnow())
        )
        await self.session.commit()
        result = await self.session.execute(select(PrescriptionGiven).where(PrescriptionGiven.id == prescription_id))
        return result.scalar_one_or_none()
