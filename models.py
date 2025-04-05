from pydantic import BaseModel, Field
from typing import List, Optional

class Step(BaseModel):
    step_id: int
    goal: str

class VerificationResult(BaseModel):
    success: bool
    message: Optional[str] = None
    new_goal: Optional[str] = None 