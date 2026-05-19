from fastapi import APIRouter

router = APIRouter(
    prefix="/profile",
    tags=["Profile"]
)

@router.get("/info")
async def get_profile_info():
    """Info channel Hippo Academy"""
    return {
        "channel_name": "Hippo Academy",
        "description": "Tech & Programming Education",
        "subscribers": 150000,
        "total_videos": 120
    }
