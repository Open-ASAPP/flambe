from flambe.compile.yaml import load_config
from flambe.metric import AUC

class TestLoadConfig:

    def test_load_config(self):
        config = """!AUC"""
        x = load_config(config)
        x = x()
        assert isinstance(x, AUC)
