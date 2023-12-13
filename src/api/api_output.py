from pydantic import BaseModel


class ApiOutput(BaseModel):
    id: int
    price_category: str
