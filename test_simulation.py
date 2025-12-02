"""
Test script for System Dynamics Backend
Run this to verify your backend is working correctly
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("Health check passed")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_simple_simulation():
    """Test simple population growth simulation"""
    print("\nTesting simple population simulation...")
    
    # Define a simple population model
    model = {
        "elements": [
            {
                "id": "pop",
                "type": "stock",
                "name": "Population",
                "initial": 1000.0,
                "equation": "births - deaths"
            },
            {
                "id": "births",
                "type": "flow",
                "name": "births",
                "equation": "birth_rate * Population"
            },
            {
                "id": "deaths",
                "type": "flow",
                "name": "deaths",
                "equation": "death_rate * Population"
            },
            {
                "id": "birth_rate",
                "type": "parameter",
                "name": "birth_rate",
                "value": 0.03
            },
            {
                "id": "death_rate",
                "type": "parameter",
                "name": "death_rate",
                "value": 0.01
            }
        ],
        "links": [
            {"id": "l1", "source": "birth_rate", "target": "births"},
            {"id": "l2", "source": "pop", "target": "births"},
            {"id": "l3", "source": "death_rate", "target": "deaths"},
            {"id": "l4", "source": "pop", "target": "deaths"}
        ],
        "config": {
            "start_time": 0.0,
            "end_time": 10.0,
            "time_step": 1.0,
            "method": "euler"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/simulate",
            json=model,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print("Simulation succeeded")
                print(f"  Initial population: {result['results']['pop'][0]:.2f}")
                print(f"  Final population: {result['results']['pop'][-1]:.2f}")
                print(f"  Time points: {len(result['time'])}")
                return True
            else:
                print(f"Simulation failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Simulation test failed: {e}")
        return False

def test_validation():
    """Test model validation endpoint"""
    print("\nTesting model validation...")
    
    model = {
        "elements": [
            {
                "id": "stock1",
                "type": "stock",
                "name": "TestStock",
                "initial": 100.0,
                "equation": "flow1"
            },
            {
                "id": "flow1",
                "type": "flow",
                "name": "TestFlow",
                "equation": "0.1 * TestStock"
            }
        ],
        "links": [
            {"id": "l1", "source": "stock1", "target": "flow1"}
        ],
        "config": {
            "start_time": 0.0,
            "end_time": 10.0,
            "time_step": 1.0,
            "method": "euler"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/validate",
            json=model,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("valid"):
                print("Validation passed")
                print(f"  Stocks: {result.get('stocks')}")
                print(f"  Flows: {result.get('flows')}")
                print(f"  Parameters: {result.get('parameters')}")
                print(f"  Variables: {result.get('variables')}")
                return True
            else:
                print(f"Validation failed: {result.get('message')}")
                return False
        else:
            print(f"HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Validation test failed: {e}")
        return False

def test_equation_features():
    """Test advanced equation features"""
    print("\nTesting advanced equation features...")
    
    model = {
        "elements": [
            {
                "id": "stock1",
                "type": "stock",
                "name": "Stock",
                "initial": 50.0,
                "equation": "rate"
            },
            {
                "id": "rate",
                "type": "flow",
                "name": "rate",
                "equation": "k * sqrt(Stock) * (100 - Stock) / 100"
            },
            {
                "id": "k",
                "type": "parameter",
                "name": "k",
                "value": 0.5
            }
        ],
        "links": [
            {"id": "l1", "source": "k", "target": "rate"},
            {"id": "l2", "source": "stock1", "target": "rate"}
        ],
        "config": {
            "start_time": 0.0,
            "end_time": 20.0,
            "time_step": 1.0,
            "method": "euler"
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/simulate",
            json=model,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print("Advanced equations work (sqrt, division, subtraction)")
                return True
            else:
                print(f"Advanced equation test failed: {result.get('error')}")
                return False
        else:
            print(f"HTTP error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Advanced equation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("System Dynamics Backend Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 4
    
    if test_health():
        tests_passed += 1
    
    if test_simple_simulation():
        tests_passed += 1
    
    if test_validation():
        tests_passed += 1
    
    if test_equation_features():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("\nAll tests passed! Backend is working correctly.")
        print("You can now use the frontend React application.")
    else:
        print("\n Some tests failed. Please check:")
        print("  1. Backend is running (python main.py)")
        print("  2. Backend is on http://localhost:8000")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
    main()
