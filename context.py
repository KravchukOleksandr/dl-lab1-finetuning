# system_context.py
import os
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SystemParams:
    config_container_name: str
    ref_data_container_name: str
    redis_user: str
    redis_stream_name: str
    keyvault_name: str
    keyvault_id: str

@dataclass(frozen=True, slots=True)
class Secrets:
    blob_connection_string: str
    redis_address: str
    redis_password: str
    bridge_user: str
    bridge_password: str
    bridge_address: str

@dataclass(frozen=True, slots=True)
class SystemContext:
    params: SystemParams
    secrets: Secrets
    url_base: str


def load_system_params_from_env() -> SystemParams:
    return SystemParams(
        config_container_name=os.environ["CONFIG_CONTAINER_NAME"],
        ref_data_container_name=os.environ["REF_DATA_CONTAINER_NAME"],
        redis_user=os.environ["REDIS_USER"],
        redis_stream_name=os.environ["REDIS_STREAM_NAME"],
        keyvault_name=os.environ["KEYVAULT_NAME"],
        keyvault_id=os.environ["KEYVAULT_ID"],
    )


def make_url_base(*, redis_user: str, redis_address: str, redis_password: str) -> str:
    # подставь вашу реальную схему
    return f"rtsp://{redis_user}:{redis_password}@{redis_address}/"


def build_system_context(params: SystemParams, secret_client) -> SystemContext:
    secrets = Secrets(
        blob_connection_string=secret_client.get("BLOB_CONNECTION_STRING"),
        redis_address=secret_client.get("REDIS_ADDRESS"),
        redis_password=secret_client.get("REDIS_PASSWORD"),
        bridge_user=secret_client.get("BRIDGE_USER"),
        bridge_password=secret_client.get("BRIDGE_PASSWORD"),
        bridge_address=secret_client.get("BRIDGE_ADDRESS"),
    )
    url_base = make_url_base(
        redis_user=params.redis_user,
        redis_address=secrets.redis_address,
        redis_password=secrets.redis_password,
    )
    return SystemContext(params=params, secrets=secrets, url_base=url_base)


def create_system_context() -> SystemContext:
    params = load_system_params_from_env()

    # тут ты создаёшь Secret Client как у тебя принято (managed identity)
    secret_client = make_secret_client(keyvault_name=params.keyvault_name, keyvault_id=params.keyvault_id)

    return build_system_context(params, secret_client)