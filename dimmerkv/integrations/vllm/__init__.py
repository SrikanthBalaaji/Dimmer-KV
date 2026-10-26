"""vLLM plugin entry point (``vllm.general_plugins``): registers ``DimmerKVConnector`` by name."""


def register() -> None:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    if "DimmerKVConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "DimmerKVConnector", "dimmerkv.integrations.vllm.connector", "DimmerKVConnector")
