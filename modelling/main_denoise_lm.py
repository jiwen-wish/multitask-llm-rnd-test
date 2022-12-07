#%%
import logging
logging.getLogger().setLevel(logging.INFO)

from main_conditional_lm import LLM
from main_utils import LLM_DenoiseData
from pytorch_lightning.cli import LightningCLI

def cli_main():
    cli = LightningCLI(LLM, LLM_DenoiseData, save_config_overwrite=True)

if __name__ == "__main__":
    cli_main()
