from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from pydantic import BaseModel
from ...core.dependencies import get_db
from ...models.user import User
from ...models.settings import UserSettings

router = APIRouter()

# Pydantic models for request/response
class UserPreferencesRequest(BaseModel):
    section: str
    timezone: str
    language: str
    notifications: Dict[str, bool]

class UserPreferencesResponse(BaseModel):
    section: str
    timezone: str
    language: str
    notifications: Dict[str, bool]

class UserAccountRequest(BaseModel):
    phone: str

class UserAccountResponse(BaseModel):
    name: str
    role: str
    email: str
    phone: str

@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    db: Session = Depends(get_db)
):
    """Get user preferences from database"""
    try:
        # For now, return default settings since we don't have authentication
        # In a real implementation, you would get the current user from the session
        return UserPreferencesResponse(
            section="JUC-LDH",  # Default section
            timezone="Asia/Kolkata",
            language="en",
            notifications={
                "train_delay_alerts": True,
                "disruption_alerts": True,
                "maintenance_alerts": False,
                "system_updates": True,
                "in_app": True,
                "email": False,
                "sms": False,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch preferences: {str(e)}")

@router.post("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences: UserPreferencesRequest,
    db: Session = Depends(get_db)
):
    """Update user preferences in database"""
    try:
        # For now, just return the preferences as-is since we don't have authentication
        # In a real implementation, you would save to database
        return UserPreferencesResponse(
            section=preferences.section,
            timezone=preferences.timezone,
            language=preferences.language,
            notifications=preferences.notifications
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")

@router.get("/account", response_model=UserAccountResponse)
async def get_user_account(
    db: Session = Depends(get_db)
):
    """Get user account information"""
    try:
        # For now, return default account info since we don't have authentication
        return UserAccountResponse(
            name="Controller User",
            role="Controller",
            email="controller@railway.com",
            phone=""
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch account: {str(e)}")

@router.post("/account", response_model=UserAccountResponse)
async def update_user_account(
    account: UserAccountRequest,
    db: Session = Depends(get_db)
):
    """Update user account information"""
    try:
        # For now, just return the account info as-is since we don't have authentication
        return UserAccountResponse(
            name="Controller User",
            role="Controller",
            email="controller@railway.com",
            phone=account.phone
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update account: {str(e)}")
