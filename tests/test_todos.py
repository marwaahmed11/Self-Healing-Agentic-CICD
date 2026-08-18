import pytest
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

# Test fixture - clear in-memory data before each test (super fast!)
@pytest.fixture(autouse=True)
def reset_db():
    """Clear all todos and reset ID counter before each test"""
    main.todos_db.clear()
    main.next_id = 1
    yield

# Health Check Tests
def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# List Todos Tests
def test_list_todos_empty():
    """Test listing todos when database is empty"""
    response = client.get("/todos")
    assert response.status_code == 200
    assert response.json() == []

def test_list_todos_with_items():
    """Test listing todos with items"""
    client.post("/todos", json={"title": "Test Todo 1", "description": "Description 1"})
    client.post("/todos", json={"title": "Test Todo 2", "description": "Description 2"})
    
    response = client.get("/todos")
    assert response.status_code == 200
    todos = response.json()
    assert len(todos) == 2
    assert todos[0]["title"] == "Test Todo 1"
    assert todos[1]["title"] == "Test Todo 2"

# Create Todo Tests
def test_create_todo_success():
    """Test creating a todo successfully"""
    response = client.post(
        "/todos",
        json={"title": "New Todo", "description": "This is a test todo", "completed": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "New Todo"
    assert data["description"] == "This is a test todo"
    assert data["completed"] == False
    assert "id" in data
    assert "created_at" in data

def test_create_todo_minimal():
    """Test creating a todo with minimal fields"""
    response = client.post("/todos", json={"title": "Simple Todo"})
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Simple Todo"
    assert data["description"] == ""

def test_create_todo_invalid_empty_title():
    """Test creating a todo with empty title fails"""
    response = client.post("/todos", json={"title": "", "description": "No title"})
    assert response.status_code == 422  # Validation error

# Get Todo Tests
def test_get_todo_success():
    """Test getting a specific todo"""
    create_response = client.post("/todos", json={"title": "Get Test", "description": "Test getting this"})
    todo_id = create_response.json()["id"]
    
    response = client.get(f"/todos/{todo_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == todo_id
    assert data["title"] == "Get Test"

def test_get_todo_not_found():
    """Test getting a non-existent todo"""
    response = client.get("/todos/999")
    assert response.status_code == 404
    assert "Todo not found" in response.json()["detail"]

# Update Todo Tests
def test_update_todo_success():
    """Test updating a todo"""
    create_response = client.post("/todos", json={"title": "Original Title", "completed": False})
    todo_id = create_response.json()["id"]
    
    response = client.put(
        f"/todos/{todo_id}",
        json={"title": "Updated Title", "completed": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["completed"] == True

def test_update_todo_partial():
    """Test partial update of a todo"""
    create_response = client.post("/todos", json={"title": "Original", "description": "Original Desc"})
    todo_id = create_response.json()["id"]
    
    response = client.put(f"/todos/{todo_id}", json={"completed": True})
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Original"  # Should remain unchanged
    assert data["description"] == "Original Desc"  # Should remain unchanged
    assert data["completed"] == True  # Should be updated

def test_update_todo_not_found():
    """Test updating a non-existent todo"""
    response = client.put("/todos/999", json={"title": "Updated"})
    assert response.status_code == 404



# Delete Todo Tests
def test_delete_todo_success():
    """Test deleting a specific todo"""
    create_response = client.post("/todos", json={"title": "To Delete"})
    todo_id = create_response.json()["id"]
    
    response = client.delete(f"/todos/{todo_id}")
    assert response.status_code == 200
    
    # Verify it's deleted
    get_response = client.get(f"/todos/{todo_id}")
    assert get_response.status_code == 404

def test_delete_todo_not_found():
    """Test deleting a non-existent todo"""
    response = client.delete("/todos/999")
    assert response.status_code == 404

def test_delete_all_todos():
    """Test deleting all todos"""
    client.post("/todos", json={"title": "Todo 1"})
    client.post("/todos", json={"title": "Todo 2"})
    
    response = client.delete("/todos")
    assert response.status_code == 200
    
    # Verify all deleted
    list_response = client.get("/todos")
    assert list_response.json() == []

# Integration Tests
def test_full_crud_workflow():
    """Test complete CRUD workflow"""
    # Create
    create_response = client.post("/todos", json={"title": "Task 1", "description": "Do something"})
    assert create_response.status_code == 200
    todo_id = create_response.json()["id"]
    
    # Read
    get_response = client.get(f"/todos/{todo_id}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Task 1"
    
    # Update
    update_response = client.put(f"/todos/{todo_id}", json={"completed": True})
    assert update_response.status_code == 200
    assert update_response.json()["completed"] == True
    
    # Delete
    delete_response = client.delete(f"/todos/{todo_id}")
    assert delete_response.status_code == 200
    
    # Verify deletion
    final_get = client.get(f"/todos/{todo_id}")
    assert final_get.status_code == 404
