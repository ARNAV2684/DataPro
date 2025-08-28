"""
Python Module Runner for Garuda ML Pipeline

This utility provides functionality to execute Python modules programmatically,
handling input/output, error capture, and result formatting for the API endpoints.
"""

import os
import sys
import subprocess
import tempfile
import importlib.util
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PythonModuleRunner:
    """Utility for running Python modules with input/output handling"""
    
    def __init__(self, workspace_root: str):
        """
        Initialize the module runner
        
        Args:
            workspace_root: Root directory of the workspace
        """
        self.workspace_root = Path(workspace_root)
        self.temp_dir = Path(tempfile.gettempdir()) / "garuda_pipeline"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Set Python executable path - prefer the data folder venv if available
        data_python = self.workspace_root / "data" / "Scripts" / "python.exe"
        if data_python.exists():
            self.python_path = str(data_python)
            logger.info(f"Using data folder virtual environment: {self.python_path}")
        else:
            self.python_path = sys.executable
            logger.info(f"Using system Python: {self.python_path}")
        
        logger.info(f"PythonModuleRunner initialized with workspace: {workspace_root}")
    
    def run_module_function(self, module_path: str, function_name: str, 
                          args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific function from a Python module
        
        Args:
            module_path: Relative path to the Python module (e.g., "2and3week/PPnumeric/scale_numeric_features.py")
            function_name: Name of the function to call
            args: Arguments to pass to the function
            
        Returns:
            Result dictionary with output, metadata, and status
        """
        try:
            # Resolve full module path
            full_module_path = self.workspace_root / module_path
            
            if not full_module_path.exists():
                return {
                    "success": False,
                    "error": f"Module not found: {module_path}",
                    "output_path": None,
                    "meta": {}
                }
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location("module", full_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if function exists
            if not hasattr(module, function_name):
                return {
                    "success": False,
                    "error": f"Function '{function_name}' not found in module {module_path}",
                    "output_path": None,
                    "meta": {}
                }
            
            # Get the function
            target_function = getattr(module, function_name)
            
            # Call the function
            result = target_function(**args)
            
            return {
                "success": True,
                "result": result,
                "output_path": args.get("output_path"),
                "meta": {"function": function_name, "module": module_path}
            }
            
        except Exception as e:
            logger.error(f"Error running module function {module_path}::{function_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output_path": None,
                "meta": {}
            }
    
    def run_module_script(self, module_path: str, input_file: str, 
                         output_file: Optional[str] = None, 
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run a Python module as a script with file input/output
        
        Args:
            module_path: Relative path to the Python module
            input_file: Path to input file
            output_file: Path to output file (auto-generated if None)
            params: Additional parameters to pass as environment variables
            
        Returns:
            Result dictionary with output path, metadata, and status
        """
        try:
            # Resolve paths
            full_module_path = self.workspace_root / module_path
            
            if not full_module_path.exists():
                return {
                    "success": False,
                    "error": f"Module not found: {module_path}",
                    "output_path": None,
                    "meta": {}
                }
            
            # Generate output file path if not provided
            if output_file is None:
                input_name = Path(input_file).stem
                output_file = str(self.temp_dir / f"{input_name}_processed.csv")
            
            # Prepare environment variables
            env = os.environ.copy()
            if params:
                for key, value in params.items():
                    env[f"PIPELINE_{key.upper()}"] = str(value)
            
            # Add workspace root to Python path for imports
            current_pythonpath = env.get('PYTHONPATH', '')
            workspace_path = str(self.workspace_root.absolute())
            if current_pythonpath:
                env['PYTHONPATH'] = f"{workspace_path}{os.pathsep}{current_pythonpath}"
            else:
                env['PYTHONPATH'] = workspace_path
            
            # Prepare command arguments with absolute paths
            cmd = [
                self.python_path,
                str(full_module_path.absolute()),
                "--input", str(Path(input_file).absolute()),
                "--output", str(Path(output_file).absolute())
            ]
            
            # Add parameter arguments
            if params:
                for key, value in params.items():
                    if isinstance(value, list):
                        # Handle list parameters (like columns)
                        cmd.append(f"--{key}")
                        cmd.extend([str(item) for item in value])
                    else:
                        cmd.extend([f"--{key}", str(value)])
            
            # Run the script from its own directory
            script_dir = full_module_path.parent
            logger.info(f"Python executable: {self.python_path}")
            logger.info(f"Working directory: {str(script_dir)}")
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(script_dir),
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_path": output_file,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "meta": {
                        "module": module_path,
                        "input_file": input_file,
                        "output_file": output_file,
                        "params": params or {}
                    }
                }
            else:
                logger.error(f"Script {module_path} failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return {
                    "success": False,
                    "error": f"Script failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_path": None,
                    "meta": {"module": module_path}
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out (5 minutes)",
                "output_path": None,
                "meta": {"module": module_path}
            }
        except Exception as e:
            logger.error(f"Error running module script {module_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output_path": None,
                "meta": {"module": module_path}
            }
    
    def adapt_legacy_module(self, module_path: str, input_file: str, 
                           params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Adapt legacy modules that use CLI prompts to programmatic input
        
        This method analyzes the module and tries to run it by providing
        automated responses to input prompts.
        
        Args:
            module_path: Relative path to the Python module
            input_file: Path to input file
            params: Parameters for the operation
            
        Returns:
            Result dictionary with output path and metadata
        """
        try:
            full_module_path = self.workspace_root / module_path
            
            if not full_module_path.exists():
                return {
                    "success": False,
                    "error": f"Module not found: {module_path}",
                    "output_path": None,
                    "meta": {}
                }
            
            # Read the module to understand its structure
            with open(full_module_path, 'r') as f:
                module_code = f.read()
            
            # Generate output file path
            input_name = Path(input_file).stem
            output_file = str(self.temp_dir / f"{input_name}_processed.csv")
            
            # Create a wrapper script that provides automated inputs
            wrapper_script = self._create_wrapper_script(
                module_path, input_file, output_file, params or {}
            )
            
            # Run the wrapper script
            result = subprocess.run(
                [sys.executable, wrapper_script],
                capture_output=True,
                text=True,
                cwd=str(self.workspace_root),
                timeout=300
            )
            
            # Clean up wrapper script
            os.unlink(wrapper_script)
            
            if result.returncode == 0 and os.path.exists(output_file):
                return {
                    "success": True,
                    "output_path": output_file,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "meta": {
                        "module": module_path,
                        "input_file": input_file,
                        "output_file": output_file,
                        "adaptation_method": "wrapper_script"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Adapted script failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_path": None,
                    "meta": {"module": module_path}
                }
                
        except Exception as e:
            logger.error(f"Error adapting legacy module {module_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": None,
                "meta": {"module": module_path}
            }
    
    def _create_wrapper_script(self, module_path: str, input_file: str, 
                              output_file: str, params: Dict[str, Any]) -> str:
        """
        Create a wrapper script for legacy modules that handles CLI prompts
        
        Args:
            module_path: Path to the original module
            input_file: Input file path
            output_file: Output file path
            params: Parameters for the operation
            
        Returns:
            Path to the created wrapper script
        """
        wrapper_content = f'''
import sys
import os
from pathlib import Path

# Add workspace root to path
workspace_root = Path(__file__).parent
sys.path.insert(0, str(workspace_root))

# Mock input function to provide automated responses
original_input = input

def mock_input(prompt=""):
    """Mock input function that provides automated responses"""
    prompt_lower = prompt.lower()
    
    # File selection prompts
    if "enter the number" in prompt_lower and "csv" in prompt_lower:
        return "1"  # Select first CSV file
    
    # Strategy selection prompts
    if "strategy" in prompt_lower or "method" in prompt_lower:
        return "{params.get('strategy', 'mean')}"
    
    # Threshold prompts
    if "threshold" in prompt_lower:
        return "{params.get('threshold', '0.5')}"
    
    # Output file prompts
    if "output" in prompt_lower and "file" in prompt_lower:
        return "{output_file}"
    
    # Default response
    return "1"

# Replace input function
__builtins__['input'] = mock_input

# Set up environment
os.chdir(str(Path("{input_file}").parent))

# Import and run the original module
try:
    exec(open("{self.workspace_root / module_path}").read())
except SystemExit:
    pass  # Ignore sys.exit() calls
'''
        
        wrapper_file = str(self.temp_dir / f"wrapper_{Path(module_path).stem}.py")
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_content)
        
        return wrapper_file
    
    def run_eda_module(self, module_path: str, input_file: str, 
                      analysis_type: str) -> Dict[str, Any]:
        """
        Specialized runner for EDA modules that generate visualizations
        
        Args:
            module_path: Path to EDA module
            input_file: Input data file
            analysis_type: Type of analysis to perform
            
        Returns:
            Result with output files and generated visualizations
        """
        try:
            # Create output directory for this analysis
            analysis_dir = self.temp_dir / f"eda_{analysis_type}_{Path(input_file).stem}"
            analysis_dir.mkdir(exist_ok=True)
            
            # Copy input file to analysis directory
            import shutil
            local_input = analysis_dir / Path(input_file).name
            shutil.copy2(input_file, local_input)
            
            # Run the module in the analysis directory
            result = self.adapt_legacy_module(
                module_path, str(local_input), {"analysis_type": analysis_type}
            )
            
            if result["success"]:
                # Find all generated files in the analysis directory
                generated_files = list(analysis_dir.glob("*"))
                visualization_files = [
                    str(f) for f in generated_files 
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.html']
                ]
                
                result["meta"]["generated_files"] = [str(f) for f in generated_files]
                result["meta"]["visualizations"] = visualization_files
                result["meta"]["analysis_directory"] = str(analysis_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running EDA module {module_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output_path": None,
                "meta": {"module": module_path, "analysis_type": analysis_type}
            }

    def run_cli_script(self, module_path: str, cli_args: List[str]) -> Dict[str, Any]:
        """
        Run a CLI script with command line arguments
        
        Args:
            module_path: Path to the Python script
            cli_args: List of command line arguments
            
        Returns:
            Result dictionary with success status and output
        """
        try:
            # Resolve the full path to the module
            full_module_path = self.workspace_root / module_path
            
            if not full_module_path.exists():
                return {
                    "success": False,
                    "error": f"Module {module_path} not found at {full_module_path}"
                }
            
            # Get the directory containing the script for proper working directory
            script_dir = full_module_path.parent
            
            # Build the command
            cmd = [str(self.python_path), str(full_module_path)] + cli_args
            
            logger.info(f"Python executable: {self.python_path}")
            logger.info(f"Working directory: {script_dir}")
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=str(script_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Script {module_path} completed successfully")
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            else:
                logger.error(f"Script {module_path} failed with return code {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return {
                    "success": False,
                    "error": f"Script failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"Script {module_path} timed out")
            return {
                "success": False,
                "error": "Script execution timed out"
            }
        except Exception as e:
            logger.error(f"Error running CLI script {module_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# ===================================
# GLOBAL INSTANCE
# ===================================

# Global runner instance
module_runner: Optional[PythonModuleRunner] = None

def get_module_runner() -> PythonModuleRunner:
    """Get the global module runner instance"""
    if module_runner is None:
        raise RuntimeError("PythonModuleRunner not initialized. Call init_module_runner() first.")
    return module_runner

def init_module_runner(workspace_root: str) -> PythonModuleRunner:
    """Initialize the global module runner instance"""
    global module_runner
    module_runner = PythonModuleRunner(workspace_root)
    return module_runner
