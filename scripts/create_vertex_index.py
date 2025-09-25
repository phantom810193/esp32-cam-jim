# scripts/create_vertex_index.py
import os
from google.cloud import aiplatform

PROJECT = os.getenv("GCP_PROJECT", "esp32cam-472912")
REGION  = os.getenv("VERTEX_REGION", "asia-east1")

def main():
    aiplatform.init(project=PROJECT, location=REGION)

    # 1) 建立 STREAM_UPDATE 索引（512 維，Cosine，Tree-AH）
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="faces-index",
        dimensions=512,
        approximate_neighbors_count=50,
        distance_measure_type="COSINE_DISTANCE",
        feature_norm_type="UNIT_L2_NORM",
        leaf_node_embedding_count=1000,
        fraction_leaf_nodes_to_search=0.05,
        shard_size="SHARD_SIZE_SMALL",
        index_update_method="STREAM_UPDATE",
        sync=True,
    )
    print("INDEX:", index.resource_name)

    # 2) 建立端點（公網）
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="faces-endpoint",
        public_endpoint_enabled=True,
        sync=True,
    )
    print("ENDPOINT:", endpoint.resource_name)

    # 3) 部署索引到端點（自訂 deployed_index_id）
    endpoint = endpoint.deploy_index(
        index=index,
        deployed_index_id="faces-deployed",
        sync=True,  # 首次部署需要配置時間，完成後才會結束
    )
    print("DEPLOYED OK")

if __name__ == "__main__":
    main()