
import os

from app import logger


def get_env_var(var_name: str, default_value, var_type=str, choices=None):
        """Get environment variable with type conversion and validation."""
        value = os.getenv(var_name)
        
        if value is None:
            return default_value
        
        # Type conversion
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            try:
                value = int(value)
            except ValueError:
                logger().warning(f"Invalid integer value for {var_name}: {value}. Using default: {default_value}")
                return default_value
        elif var_type == str:
            pass  # No conversion needed
        
        # Validate choices if provided
        if choices and value not in choices:
            logger().warning(f"Invalid choice for {var_name}: {value}. Valid choices: {choices}. Using default: {default_value}")
            return default_value
        
        return value
    