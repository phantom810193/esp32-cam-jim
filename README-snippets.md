# Pipeline 快速操作指令

```bash
python3 pipeline/compile.py
python3 pipeline/run_pipeline.py
```

執行成功後可前往 Cloud Console → Vertex AI → Pipelines，點選 `faces-embed-upsert` 觀看 Job 詳細資訊與 Logs。若需在 CLI 追蹤，使用：

```bash
gcloud ai pipeline-jobs list --region=asia-east1 --project=esp32cam-472912
```
