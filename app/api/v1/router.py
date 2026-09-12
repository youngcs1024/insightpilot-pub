"""Versioned business routes are added by their owning roadmap steps."""

from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.conversations import router as conversations_router

router = APIRouter()
router.include_router(auth_router)

router.include_router(conversations_router)
router.include_router(chat_router)
