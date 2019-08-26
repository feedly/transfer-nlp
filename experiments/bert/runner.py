from experiments.bert.bert import *
from transfer_nlp.plugins.config import ExperimentConfig

from ..utils import PLUGINS

logger = logging.getLogger(__name__)

for plugin_name, plugin in PLUGINS.items():
    register_plugin(registrable=plugin, alias=plugin_name)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    home_env = str(Path.home() / 'work/transfer-nlp-data')

    path = './bert.json'
    experiment = ExperimentConfig(path, HOME=home_env)
    experiment.experiment['trainer'].train()
