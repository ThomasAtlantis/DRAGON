import json
import sys
from datetime import datetime
from pathlib import Path

from dragon.config import DragonConfig
from dragon.utils.mlogging import Logger


class Evaluator:
    def __init__(self, config: DragonConfig, name: str = None):
        self.logger = Logger.build(__class__.__name__, level="INFO")
        self.logger.info(f"Evaluation of `{name}` starts.")
        self.run_id = name + "-" + datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = config.evaluator
        self.config_dict = config.as_dict()
        self.name = name

    def save_output(self, output):
        run_output_dir = Path(self.config.output_dir, self.run_id)
        run_output_dir.mkdir(parents=True, exist_ok=True)
        script = " ".join(["python", "-u"] + sys.argv)
        script = script.replace("--", "\\\n    --")

        with open(run_output_dir / "output.json", "w") as f:
            json.dump(output, f)
        with open(run_output_dir / "config.json", "w") as f:
            json.dump(self.config_dict, f)
        with open(run_output_dir / "run.sh", "w") as f:
            f.write(script)
        self.logger.info(f"Output saved to `{run_output_dir}`.")

