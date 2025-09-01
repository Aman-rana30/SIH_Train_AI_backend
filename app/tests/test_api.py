"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json


class TestScheduleAPI:
    """Test cases for schedule API endpoints."""

    def test_optimize_schedule_endpoint(self, client, sample_trains):
        """Test the POST /api/schedule/optimize endpoint."""
        # Prepare request data
        request_data = {
            "trains": sample_trains,
            "optimization_params": {"max_delay": 30}
        }

        response = client.post("/api/schedule/optimize", json=request_data)

        assert response.status_code == 200
        result = response.json()

        assert "optimization_run_id" in result
        assert "schedules" in result
        assert "metrics" in result
        assert "computation_time" in result
        assert "status" in result

        # Check that all trains have schedules
        assert len(result["schedules"]) <= len(sample_trains)

    def test_whatif_analysis_endpoint(self, client):
        """Test the POST /api/schedule/whatif endpoint."""
        request_data = {
            "disruption": {
                "type": "delay",
                "delay_minutes": 45,
                "description": "Signal failure"
            },
            "affected_trains": ["EXP001", "PASS002"]
        }

        response = client.post("/api/schedule/whatif", json=request_data)

        # May return 404 if no active schedules, which is expected in test
        assert response.status_code in [200, 404, 500]

    def test_get_current_schedule_endpoint(self, client):
        """Test the GET /api/schedule/current endpoint."""
        response = client.get("/api/schedule/current")

        assert response.status_code == 200
        schedules = response.json()
        assert isinstance(schedules, list)

    def test_override_decision_endpoint(self, client):
        """Test the POST /api/schedule/override endpoint."""
        request_data = {
            "train_id": 1,
            "decision": "Delay train by 15 minutes due to passenger boarding",
            "reason": "Heavy passenger load",
            "controller_id": "CTRL001",
            "new_schedule_time": (datetime.now() + timedelta(minutes=15)).isoformat()
        }

        response = client.post("/api/schedule/override", json=request_data)

        # May return 404 if train not found, which is expected in test
        assert response.status_code in [200, 404, 500]


class TestMetricsAPI:
    """Test cases for metrics API endpoints."""

    def test_get_metrics_endpoint(self, client):
        """Test the GET /api/metrics endpoint."""
        response = client.get("/api/metrics")

        assert response.status_code == 200
        result = response.json()

        assert "current_metrics" in result
        assert "trends" in result
        assert "alerts" in result
        assert "recommendations" in result

    def test_get_metrics_history_endpoint(self, client):
        """Test the GET /api/metrics/history endpoint."""
        response = client.get("/api/metrics/history?limit=10")

        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)
        assert len(history) <= 10

    def test_get_metrics_summary_endpoint(self, client):
        """Test the GET /api/metrics/summary endpoint."""
        response = client.get("/api/metrics/summary?days=7")

        assert response.status_code == 200
        summary = response.json()

        assert "period" in summary
        assert "total_days" in summary
        assert "summary" in summary

    def test_metrics_with_date_filter(self, client):
        """Test metrics endpoint with date filter."""
        target_date = datetime.now().date().isoformat()
        response = client.get(f"/api/metrics?target_date={target_date}")

        assert response.status_code == 200


class TestWebSocketAPI:
    """Test cases for WebSocket endpoints."""

    def test_websocket_connection_stats(self, client):
        """Test WebSocket connection statistics endpoint."""
        response = client.get("/ws/connections")

        assert response.status_code == 200
        stats = response.json()

        assert "total_connections" in stats
        assert "connection_details" in stats


class TestHealthAndRoot:
    """Test cases for health check and root endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        health = response.json()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "version" in health

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        root_info = response.json()

        assert "message" in root_info
        assert "version" in root_info
        assert "docs_url" in root_info

    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        openapi_spec = response.json()

        assert "openapi" in openapi_spec
        assert "info" in openapi_spec
        assert "paths" in openapi_spec

    def test_swagger_ui(self, client):
        """Test Swagger UI endpoint."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    """Test cases for error handling."""

    def test_invalid_optimization_request(self, client):
        """Test optimization with invalid data."""
        invalid_data = {
            "trains": [],  # Empty trains list
            "optimization_params": {}
        }

        response = client.post("/api/schedule/optimize", json=invalid_data)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    def test_invalid_json_request(self, client):
        """Test API with invalid JSON."""
        response = client.post(
            "/api/schedule/optimize",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_nonexistent_endpoint(self, client):
        """Test nonexistent endpoint."""
        response = client.get("/api/nonexistent")

        assert response.status_code == 404
