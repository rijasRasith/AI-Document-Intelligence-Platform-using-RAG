import os
import sys
import time
import json
import unittest
from fastapi.testclient import TestClient

from main import app, Base, engine, SessionLocal


class TestAIDocumentAssistantRAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)
        cls.email = f"test_{int(time.time())}@example.com"
        cls.password = "SecretPassword123!"
        cls.token = None
        cls.doc_id = None

    def test_01_register_user(self):
        response = self.client.post("/api/auth/register", json={"email": self.email, "password": self.password})
        self.assertIn(response.status_code, [201, 400])
        if response.status_code == 201:
            data = response.json()
            self.assertIn("access_token", data)
            TestAIDocumentAssistantRAG.token = data["access_token"]

    def test_02_login_user(self):
        response = self.client.post("/api/auth/login", json={"email": self.email, "password": self.password})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("access_token", data)
        TestAIDocumentAssistantRAG.token = data["access_token"]

    def test_03_upload_document(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        sample_text = (
            "Annual Business Performance Report 2025.\n"
            "Section 1: Financial Highlights.\n"
            "Total company revenue grew by 28% year-over-year reaching $18.5 Billion in fiscal year 2025.\n"
            "Operating cash flow increased by 35% with gross profit margins expanding to 62%.\n"
            "Section 2: Key Risks.\n"
            "1. Supply chain bottleneck in Asian logistics hubs.\n"
            "2. Currency exchange rate fluctuations in European markets.\n"
            "Section 3: Action Items.\n"
            "Action 1: Review operational expenses by Q3. Owner: CFO Office. Priority: High.\n"
            "Action 2: Expand international marketing campaigns. Owner: Global Strategy Team. Priority: Medium."
        )
        files = {"file": ("Annual_Report_2025.txt", sample_text.encode("utf-8"), "text/plain")}
        response = self.client.post("/api/documents/upload", headers=headers, files=files)
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertIn("document_id", data)
        TestAIDocumentAssistantRAG.doc_id = data["document_id"]

    def test_04_list_documents(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        time.sleep(1.0)
        response = self.client.get("/api/documents", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data) > 0)

    def test_05_chat_rag_query(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        payload = {
            "question": "What was the total company revenue in 2025?",
            "document_ids": [TestAIDocumentAssistantRAG.doc_id],
            "top_k": 3
        }
        response = self.client.post("/api/chat", headers=headers, json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)

    def test_06_document_summary(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        response = self.client.post(f"/api/documents/{TestAIDocumentAssistantRAG.doc_id}/summary", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)

    def test_07_document_insights(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        response = self.client.post(f"/api/documents/{TestAIDocumentAssistantRAG.doc_id}/insights", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("insights", data)

    def test_08_document_action_items(self):
        headers = {"Authorization": f"Bearer {TestAIDocumentAssistantRAG.token}"}
        response = self.client.post(f"/api/documents/{TestAIDocumentAssistantRAG.doc_id}/action-items", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("action_items", data)

    def test_09_evaluate_rag(self):
        payload = {
            "retrieved_chunk_ids": ["c1", "c2", "c3"],
            "ground_truth_chunk_ids": ["c1", "c3"],
            "answer_text": "Total revenue grew 28% to $18.5 Billion in 2025.",
            "context_texts": ["Total company revenue grew by 28% year-over-year reaching $18.5 Billion in fiscal year 2025."]
        }
        response = self.client.post("/api/evaluate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("retrieval_evaluation", data)
        self.assertIn("generation_evaluation", data)


if __name__ == "__main__":
    unittest.main()
