"""End-to-end Python client example interacting with SpectrumBloom REST API."""

from __future__ import annotations

from pathlib import Path

import httpx


def main() -> None:
    base_url = "http://127.0.0.1:8000"
    dataset_path = Path("examples/sample_ecommerce.csv")

    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        health = client.get("/health").json()
        print("Health:", health)

        session = client.post(
            "/api/v1/sessions",
            json={"provider": "ollama", "model": "qwen3:4b-instruct"},
        ).json()
        sid = session["session_id"]
        print("Created session:", sid)

        with dataset_path.open("rb") as f:
            eda = client.post(
                f"/api/v1/sessions/{sid}/upload",
                files={"file": (dataset_path.name, f, "text/csv")},
            ).json()
        print(
            f"Profiled {eda['profile']['rows']} rows x {eda['profile']['columns']} cols; "
            f"generated {len(eda['figures'])} automatic Plotly charts."
        )

        ml = client.post(
            f"/api/v1/sessions/{sid}/ml/experiment",
            json={"target_column": "churned", "task_type": "classification", "cv_folds": 5},
        ).json()
        print("ML Best Model:", ml["best_model"], ml["models"])

        query_res = client.post(
            f"/api/v1/sessions/{sid}/query",
            json={
                "query": "Which category has the highest mean revenue, and create a Plotly bar chart of total revenue by category?"
            },
        ).json()
        print("Agent Answer:\n", query_res["answer"])
        print("Figures generated:", len(query_res["figures"]))
        print("Evaluation:", query_res["evaluation"])


if __name__ == "__main__":
    main()
