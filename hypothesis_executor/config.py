"""Configuration for the Hypothesis Executor Agent."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutorConfig:
    """Configuration for GLM-5 API, proxy, and generation parameters."""

    # GLM-5 API
    glm_api_key: str = "c1e79ac1c5bf4ba3b1bff2c6c993a89d.xZPJIuUb1iqOGLt7"
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-5"
    glm_max_tokens: int = 65536
    glm_temperature: float = 1.0
    glm_enable_thinking: bool = True

    # Proxy
    proxy_host: str = "82.38.13.226"
    proxy_port: int = 46352
    proxy_user: str = "JMMW42KD"
    proxy_pass: str = "ISK9OAWD"
    use_proxy: bool = False

    # Retry
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    experiments_dir: str = "experiments"
    models_dir: str = "models"

    # Generation
    max_generation_attempts: int = 2

    @property
    def proxy_url(self) -> str:
        return (
            f"http://{self.proxy_user}:{self.proxy_pass}"
            f"@{self.proxy_host}:{self.proxy_port}"
        )
