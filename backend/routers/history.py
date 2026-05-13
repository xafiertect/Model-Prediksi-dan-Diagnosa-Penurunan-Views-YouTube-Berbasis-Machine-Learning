from fastapi import APIRouter

router = APIRouter(
    prefix="/history",
    tags=["History"]
)

@router.get("/")
async def get_history():
    """
    Skeleton endpoint for fetching prediction history.
    To be implemented with database integration later.
    """
    return {"message": "History endpoint is working", "data": []}
