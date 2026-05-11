from fastapi import APIRouter

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"]
)

@router.get("/")
async def get_stats():
    """
    Skeleton endpoint for fetching aggregate statistics.
    To be implemented with database integration later.
    """
    return {"message": "Stats endpoint is working", "data": {}}
