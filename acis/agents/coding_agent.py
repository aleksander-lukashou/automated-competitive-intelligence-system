"""
Coding Agent

This module implements the coding agent for analyzing competitor performance data
by generating and executing code snippets.
"""

import asyncio
import json
import logging
import tempfile
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import base64
from io import BytesIO

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class CodingAgent:
    """Agent for generating and executing code to analyze competitor data."""

    def __init__(self):
        """Initialize the coding agent."""
        self.llm = OpenAI(
            temperature=0.1,  # Lower temperature for more deterministic code generation
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Define code generation prompt
        self.code_prompt = PromptTemplate(
            input_variables=["task", "data_description", "output_format"],
            template="""
            Generate Python code to analyze the following competitor data:
            
            Data Description:
            {data_description}
            
            Task:
            {task}
            
            Output Format:
            {output_format}
            
            The code should:
            1. Be correct, efficient, and well-commented
            2. Handle potential errors gracefully
            3. Use pandas, numpy, and matplotlib as needed
            4. Return results in the requested format
            
            Only provide the Python code without any additional explanation. 
            The code will be executed directly, so ensure it's complete and runnable.
            """
        )
        
        self.code_chain = LLMChain(llm=self.llm, prompt=self.code_prompt)
        
        # Define code improvement prompt
        self.improvement_prompt = PromptTemplate(
            input_variables=["code", "error", "improvement_goal"],
            template="""
            The following Python code needs improvement:
            
            ```python
            {code}
            ```
            
            Error or Issue:
            {error}
            
            Improvement Goal:
            {improvement_goal}
            
            Please provide a corrected version of the code. Only provide the Python code without any explanation.
            """
        )
        
        self.improvement_chain = LLMChain(llm=self.llm, prompt=self.improvement_prompt)

    async def analyze_financial_data(self, 
                                     competitor: str, 
                                     financial_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze competitor financial data using generated code.
        
        Args:
            competitor: Competitor name
            financial_data: List of financial metrics with historical data
            
        Returns:
            Analysis results with insights and visualizations
        """
        try:
            # Prepare data description
            data_description = f"Financial metrics for {competitor} including: "
            data_description += ", ".join([metric.get("name", "Unknown") for metric in financial_data[:5]])
            if len(financial_data) > 5:
                data_description += f", and {len(financial_data) - 5} more metrics."
            
            # Define the analysis task
            task = f"""
            1. Calculate key financial ratios and growth rates
            2. Identify trends and patterns in the data
            3. Highlight anomalies or significant changes
            4. Generate visualizations of key metrics over time
            5. Provide a summary of financial performance
            """
            
            # Define output format
            output_format = """
            Return a JSON structure with the following keys:
            - 'summary': A text summary of the analysis
            - 'metrics': Dictionary of calculated metrics
            - 'trends': List of identified trends with supporting data
            - 'anomalies': List of anomalies with supporting data
            - 'visualizations': Dictionary of base64-encoded PNG images of plots
            
            For visualizations, use plt.figure() to create each plot, then save it to a BytesIO object and encode as base64.
            """
            
            # Generate analysis code
            code = await self._generate_code(
                task=task,
                data_description=data_description,
                output_format=output_format
            )
            
            # Execute the code
            results = await self._execute_code(code, {"data": financial_data})
            
            # Process and return results
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing financial data: {e}")
            return {
                "summary": f"Error analyzing {competitor} financial data: {str(e)}",
                "metrics": {},
                "trends": [],
                "anomalies": [],
                "visualizations": {}
            }

    async def analyze_competitive_positioning(self, 
                                             competitor: str,
                                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competitive positioning using generated code.
        
        Args:
            competitor: Competitor name
            market_data: Market position data including competitors
            
        Returns:
            Analysis results with insights and visualizations
        """
        try:
            # Prepare data description
            data_description = f"Market position data for {competitor} and competitors. " 
            data_description += f"Includes market share, strengths, weaknesses, and segment data."
            
            # Define the analysis task
            task = f"""
            1. Compare {competitor}'s market share against competitors
            2. Analyze strengths and weaknesses relative to the market
            3. Identify key differentiators and competitive advantages
            4. Generate a competitive positioning visualization
            5. Provide a summary of competitive positioning
            """
            
            # Define output format
            output_format = """
            Return a JSON structure with the following keys:
            - 'summary': A text summary of the analysis
            - 'position_score': Overall positioning score (0-100)
            - 'strengths': Ranked list of key strengths with scores
            - 'weaknesses': Ranked list of key weaknesses with scores
            - 'opportunities': List of potential opportunities based on the analysis
            - 'threats': List of potential threats based on the analysis
            - 'visualizations': Dictionary of base64-encoded PNG images of plots
            """
            
            # Generate analysis code
            code = await self._generate_code(
                task=task,
                data_description=data_description,
                output_format=output_format
            )
            
            # Execute the code
            results = await self._execute_code(code, {"data": market_data})
            
            # Process and return results
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing competitive positioning: {e}")
            return {
                "summary": f"Error analyzing {competitor} competitive positioning: {str(e)}",
                "position_score": 0,
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "threats": [],
                "visualizations": {}
            }

    async def forecast_performance(self, 
                                  competitor: str,
                                  historical_data: Dict[str, Any],
                                  forecast_period: int = 4) -> Dict[str, Any]:
        """
        Generate forecasts of future performance using historical data.
        
        Args:
            competitor: Competitor name
            historical_data: Historical performance data
            forecast_period: Number of periods to forecast
            
        Returns:
            Forecast results with visualizations
        """
        try:
            # Prepare data description
            data_description = f"Historical performance data for {competitor} including "
            if "metrics" in historical_data:
                data_description += f"{len(historical_data['metrics'])} metrics over "
            data_description += f"{forecast_period} periods to forecast ahead."
            
            # Define the analysis task
            task = f"""
            1. Perform time series analysis on the historical data
            2. Choose appropriate forecasting methods based on the data patterns
            3. Generate forecasts for key metrics for the next {forecast_period} periods
            4. Provide confidence intervals for the forecasts
            5. Create visualizations showing historical data and forecasts
            """
            
            # Define output format
            output_format = """
            Return a JSON structure with the following keys:
            - 'summary': A text summary of the forecast
            - 'forecasts': Dictionary of forecasted values for each metric
            - 'confidence_intervals': Dictionary of upper and lower bounds for each forecast
            - 'methods': Description of forecasting methods used
            - 'accuracy': Estimated accuracy metrics of the forecast
            - 'visualizations': Dictionary of base64-encoded PNG images of plots
            """
            
            # Generate analysis code
            code = await self._generate_code(
                task=task,
                data_description=data_description,
                output_format=output_format
            )
            
            # Execute the code
            results = await self._execute_code(code, {"data": historical_data, "periods": forecast_period})
            
            # Process and return results
            return results
            
        except Exception as e:
            logger.error(f"Error forecasting performance: {e}")
            return {
                "summary": f"Error forecasting {competitor} performance: {str(e)}",
                "forecasts": {},
                "confidence_intervals": {},
                "methods": "Failed to determine appropriate forecasting methods",
                "accuracy": {},
                "visualizations": {}
            }

    async def _generate_code(self, task: str, data_description: str, output_format: str) -> str:
        """
        Generate code for a specific analysis task.
        
        Args:
            task: Description of the analysis task
            data_description: Description of the data structure
            output_format: Expected output format
            
        Returns:
            Generated Python code
        """
        try:
            # Generate code using LLM
            code = await asyncio.to_thread(
                self.code_chain.run,
                task=task,
                data_description=data_description,
                output_format=output_format
            )
            
            # Clean up and return code
            return code.strip()
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            # Return a basic error handling code snippet
            return """
            def analyze(data):
                try:
                    return {
                        "summary": "Code generation failed, unable to perform analysis",
                        "error": "Error in code generation"
                    }
                except Exception as e:
                    return {"error": str(e)}
            
            result = analyze(data)
            """

    async def _execute_code(self, code: str, inputs: Dict[str, Any], 
                           max_retries: int = 2) -> Dict[str, Any]:
        """
        Execute generated code in a secure environment.
        
        Args:
            code: Python code to execute
            inputs: Input data for the code
            max_retries: Maximum number of improvement attempts
            
        Returns:
            Results from code execution
        """
        retries = 0
        while retries <= max_retries:
            try:
                # Create a temporary directory for execution
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save inputs as JSON
                    input_path = os.path.join(temp_dir, "inputs.json")
                    with open(input_path, "w") as f:
                        json.dump(inputs, f)
                    
                    # Create a wrapper script that loads inputs and runs the generated code
                    script_path = os.path.join(temp_dir, "analysis.py")
                    with open(script_path, "w") as f:
                        f.write("""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sys

# Load inputs
with open("inputs.json", "r") as f:
    inputs = json.load(f)

# Make inputs available to the generated code
locals().update(inputs)

# Utility function for encoding plots
def encode_plot(fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Execute generated code
try:
""")
                        # Indent the generated code
                        for line in code.split("\n"):
                            f.write(f"    {line}\n")
                        
                        # Add code to output the results
                        f.write("""
    # Ensure result is JSON serializable
    if 'result' not in locals():
        result = {"error": "Code did not generate a 'result' variable"}
    
    # Output results as JSON
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e), "traceback": str(sys.exc_info())}))
""")
                    
                    # Execute the script
                    process = await asyncio.create_subprocess_exec(
                        "python", script_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # Get output
                    stdout, stderr = await process.communicate()
                    
                    # Check for errors
                    if process.returncode != 0:
                        error_msg = stderr.decode().strip()
                        logger.error(f"Code execution failed: {error_msg}")
                        
                        if retries < max_retries:
                            # Try to improve the code
                            code = await self._improve_code(
                                code=code,
                                error=error_msg,
                                improvement_goal="Fix execution errors"
                            )
                            retries += 1
                            continue
                        else:
                            return {
                                "error": f"Code execution failed after {max_retries} attempts",
                                "details": error_msg
                            }
                    
                    # Parse the output
                    try:
                        output = stdout.decode().strip()
                        result = json.loads(output)
                        return result
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON output: {output}")
                        return {"error": "Failed to parse results", "raw_output": output}
                    
            except Exception as e:
                logger.error(f"Error executing code: {e}")
                if retries < max_retries:
                    # Try to improve the code
                    code = await self._improve_code(
                        code=code,
                        error=str(e),
                        improvement_goal="Fix execution environment errors"
                    )
                    retries += 1
                else:
                    return {"error": f"Code execution failed: {str(e)}"}
        
        # If we got here, all retries failed
        return {"error": "Maximum retries exceeded, unable to execute code successfully"}

    async def _improve_code(self, code: str, error: str, improvement_goal: str) -> str:
        """
        Improve problematic code using LLM.
        
        Args:
            code: Original code
            error: Error message or issue description
            improvement_goal: Goal for the improvement
            
        Returns:
            Improved code
        """
        try:
            # Generate improved code using LLM
            improved_code = await asyncio.to_thread(
                self.improvement_chain.run,
                code=code,
                error=error,
                improvement_goal=improvement_goal
            )
            
            # Return improved code
            return improved_code.strip()
            
        except Exception as e:
            logger.error(f"Error improving code: {e}")
            # Return original code if improvement fails
            return code 