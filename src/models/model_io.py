import os
import json
from typing import Dict, Any, Optional

from src.ops.event_logger import get_logger

logger = get_logger()

class ModelBundleManager:
    """
    Handles saving and loading of the trained Phase C LightGBM model
    along with its necessary feature metadata (like fracdiff values).
    """
    def __init__(self, model_dir: str = "models", model_name: str = "meta_model_v1"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, f"{model_name}.txt")
        self.meta_path = os.path.join(model_dir, f"{model_name}_meta.json")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_bundle(self, model: Any, metadata: Dict[str, Any]) -> bool:
        """
        Saves the LightGBM booster and its JSON metadata.
        """
        try:
            model.save_model(self.model_path)
            
            with open(self.meta_path + '.tmp', 'w') as f:
                json.dump(metadata, f, indent=2)
            os.replace(self.meta_path + '.tmp', self.meta_path)
            
            logger.info("model_bundle_saved", {"path": self.model_path})
            return True
        except Exception as e:
            logger.error("model_bundle_save_failed", {"error": str(e)})
            return False
            
    def load_model(self) -> Any:
        """Loads the trained LightGBM booster."""
        if not os.path.exists(self.model_path):
            return None
            
        try:
            import lightgbm as lgb
            return lgb.Booster(model_file=self.model_path)
        except Exception as e:
            logger.error("model_load_failed", {"error": str(e)})
            return None
            
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads model metadata (e.g. required features, optimal fractional diffs)."""
        if not os.path.exists(self.meta_path):
            return None
            
        try:
            with open(self.meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error("metadata_load_failed", {"error": str(e)})
            return None
