"""
Write-Audit-Publish (WAP) Utilities

Implements atomic write pattern for data integrity.
All parquet writes should use these functions.

Pattern:
1. Write to temporary file
2. Audit (read back and verify)
3. Atomic rename to final location
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import numpy as np

from src.ops.event_logger import get_logger

logger = get_logger()


def write_parquet_wap(
    df: pd.DataFrame,
    path: Union[str, Path],
    audit: bool = True,
    **parquet_kwargs
) -> Path:
    """
    Write DataFrame to parquet using Write-Audit-Publish pattern.
    
    Args:
        df: DataFrame to write
        path: Final destination path
        audit: Whether to verify write before publishing
        **parquet_kwargs: Additional arguments for to_parquet()
        
    Returns:
        Path to written file
        
    Raises:
        RuntimeError: If audit fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    temp_path = path.with_suffix('.tmp')
    
    try:
        # Step 1: Write
        df.to_parquet(temp_path, index=False, **parquet_kwargs)
        
        # Step 2: Audit (if enabled)
        if audit:
            _audit_parquet(temp_path, df)
        
        # Step 3: Publish (atomic rename)
        # On Unix, rename is atomic if destination doesn't exist
        # On Windows, may need to remove first
        if path.exists():
            path.unlink()
        temp_path.rename(path)
        
        logger.debug("parquet_write_complete", {
            "path": str(path),
            "rows": len(df),
            "cols": len(df.columns),
            "audited": audit
        })
        
        return path
        
    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        logger.error("parquet_write_failed", {
            "path": str(path),
            "error": str(e)
        })
        raise


def _audit_parquet(temp_path: Path, original_df: pd.DataFrame) -> None:
    """
    Audit a parquet file by reading it back and comparing with original.
    
    Args:
        temp_path: Path to temporary parquet file
        original_df: Original DataFrame for comparison
        
    Raises:
        RuntimeError: If audit fails
    """
    try:
        # Read back
        audit_df = pd.read_parquet(temp_path)
        
        # Check row count
        if len(audit_df) != len(original_df):
            raise RuntimeError(
                f"Row count mismatch: expected {len(original_df)}, got {len(audit_df)}"
            )
        
        # Check column names (order may differ)
        if set(audit_df.columns) != set(original_df.columns):
            missing = set(original_df.columns) - set(audit_df.columns)
            extra = set(audit_df.columns) - set(original_df.columns)
            raise RuntimeError(
                f"Column mismatch: missing={missing}, extra={extra}"
            )
        
        # Check for null count changes (indicates data loss)
        for col in original_df.columns:
            orig_nulls = original_df[col].isna().sum()
            audit_nulls = audit_df[col].isna().sum()
            if orig_nulls != audit_nulls:
                raise RuntimeError(
                    f"Null count mismatch in '{col}': expected {orig_nulls}, got {audit_nulls}"
                )
        
        logger.debug("parquet_audit_passed", {
            "path": str(temp_path),
            "rows": len(audit_df)
        })
        
    except Exception as e:
        raise RuntimeError(f"Parquet audit failed: {e}")


def read_parquet_safe(path: Union[str, Path]) -> pd.DataFrame:
    """
    Safely read a parquet file with error handling.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If read fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    # Check for temp file (indicates interrupted write)
    temp_path = path.with_suffix('.tmp')
    if temp_path.exists():
        logger.warn("temp_file_detected", {
            "temp": str(temp_path),
            "final": str(path)
        })
        # Remove stale temp file
        temp_path.unlink()
    
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet {path}: {e}")
