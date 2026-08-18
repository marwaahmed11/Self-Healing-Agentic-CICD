from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import List, Optional

# Pydantic Models
class TodoCreate(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = ""
    completed: bool = False

class TodoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    completed: Optional[bool] = None

class Todo(BaseModel):
    id: int
    title: str
    description: str
    completed: bool
    created_at: datetime

# FastAPI App
app = FastAPI(title="Todo API", version="1.0.0")

# In-memory storage
todos_db: dict[int, dict] = {}
next_id = 1

# CRUD Endpoints
@app.get("/todos", response_model=List[Todo])
def list_todos():
    """Get all todos"""
    return [
        Todo(
            id=id,
            title=todo["title"],
            description=todo["description"],
            completed=todo["completed"],
            created_at=todo["created_at"]
        )
        for id, todo in sorted(todos_db.items())
    ]

@app.post("/todos", response_model=Todo)
def create_todo(todo: TodoCreate):
    """Create a new todo"""
    global next_id
    todo_id = next_id
    next_id += 1
    
    todo_data = {
        "id": todo_id,
        "title": todo.title,
        "description": todo.description,
        "completed": todo.completed,
        "created_at": datetime.now(timezone.utc)
    }
    todos_db[todo_id] = todo_data
    return Todo(**todo_data)

@app.get("/todos/{todo_id}", response_model=Todo)
def get_todo(todo_id: int):
    """Get a specific todo by ID"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    todo = todos_db[todo_id]
    return Todo(**todo)

@app.put("/todos/{todo_id}", response_model=Todo)
def update_todo(todo_id: int, todo_update: TodoUpdate):
    """Update a specific todo"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    todo = todos_db[todo_id]
    update_data = todo_update.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            todo[key] = value
    
    return Todo(**todo)

@app.delete("/todos/{todo_id}")
def delete_todo(todo_id: int):
    """Delete a specific todo"""
    if todo_id not in todos_db:
        raise HTTPException(status_code=404, detail="Todo not found")
    
    del todos_db[todo_id]
    return {"message": "Todo deleted successfully"}

@app.delete("/todos")
def delete_all_todos():
    """Delete all todos"""
    todos_db.clear()
    return {"message": "All todos deleted successfully"}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}