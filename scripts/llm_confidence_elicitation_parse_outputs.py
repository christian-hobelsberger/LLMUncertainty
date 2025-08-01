import pandas as pd
import json
import re
import yaml
import os
import argparse
from pathlib import Path
from typing import Dict, List, Union, Tuple
from tqdm import tqdm


class LLMOutputParser:
    """
    A comprehensive parser for LLM outputs that can extract answers and confidence scores
    from various response formats including JSON, YAML, and natural language patterns.
    """
    
    def __init__(self, use_llm_for_parsing: bool = False, model_wrapper=None):
        self.use_llm_for_parsing = use_llm_for_parsing
        self.model_wrapper = model_wrapper
        
    def clean_and_parse_json(self, output: str) -> Dict[str, Union[str, int, None]]:
        """
        Parse JSON from LLM output with multiple fallback strategies.
        """
        try:
            output = output.strip()
            
            # Handle empty or minimal outputs
            if not output or len(output) <= 1:
                return {"answer": None, "confidence": None}
            
            # Strategy 1: Find complete JSON objects
            json_matches = re.findall(r'\{[^{}]*\}', output, re.DOTALL)
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and any(key in parsed for key in ['answer', 'confidence']):
                        return {
                            "answer": parsed.get("answer"),
                            "confidence": self._clean_confidence(parsed.get("confidence"))
                        }
                except json.JSONDecodeError:
                    continue
            
            # Strategy 2: Find nested JSON objects (handle multi-line responses)
            nested_matches = re.findall(r'\{.*?\}', output, re.DOTALL)
            for match in nested_matches:
                try:
                    # Clean up common formatting issues
                    cleaned_match = re.sub(r'(?<!")(\w+)(?=":)', r'"\1', match)  # Add quotes to keys
                    cleaned_match = re.sub(r':\s*([^",\{\}\[\]]+)(?=\s*[,\}])', r': "\1"', cleaned_match)  # Quote values
                    parsed = json.loads(cleaned_match)
                    if isinstance(parsed, dict):
                        return {
                            "answer": parsed.get("answer"),
                            "confidence": self._clean_confidence(parsed.get("confidence"))
                        }
                except (json.JSONDecodeError, re.error):
                    continue
            
            # Strategy 3: Try YAML parsing
            yaml_result = self._try_yaml_parse(output)
            if yaml_result["answer"] is not None or yaml_result["confidence"] is not None:
                return yaml_result
            
            # Strategy 4: Pattern-based extraction
            pattern_result = self._extract_with_patterns(output)
            if pattern_result["answer"] is not None or pattern_result["confidence"] is not None:
                return pattern_result
            
        except Exception as e:
            print(f"⚠️ Error in JSON parsing: {e}")
        
        return {"answer": None, "confidence": None}
    
    def _try_yaml_parse(self, output: str) -> Dict[str, Union[str, int, None]]:
        """Try to parse as YAML format."""
        try:
            # Look for YAML-like patterns
            yaml_lines = []
            for line in output.split('\n'):
                line = line.strip()
                if ':' in line and (
                    'answer' in line.lower() or 
                    'confidence' in line.lower()
                ):
                    yaml_lines.append(line)
            
            if yaml_lines:
                yaml_text = '\n'.join(yaml_lines)
                parsed = yaml.safe_load(yaml_text)
                if isinstance(parsed, dict):
                    return {
                        "answer": parsed.get("answer"),
                        "confidence": self._clean_confidence(parsed.get("confidence"))
                    }
        except Exception:
            pass
        return {"answer": None, "confidence": None}
    
    def _extract_with_patterns(self, output: str) -> Dict[str, Union[str, int, None]]:
        """Extract answer and confidence using regex patterns."""
        result = {"answer": None, "confidence": None}
        
        # Pattern 1: "Answer and Confidence (0-100): [answer], [confidence]%"
        vanilla_pattern = r'Answer and Confidence \(0-100\):\s*([^,]+),\s*(\d+)%'
        match = re.search(vanilla_pattern, output, re.IGNORECASE)
        if match:
            return {
                "answer": match.group(1).strip(),
                "confidence": int(match.group(2))
            }
        
        # Pattern 2: Look for True/False answers
        bool_patterns = [
            r'\b(True|False)\b.*?(\d{1,3})',
            r'answer["\']?\s*:\s*["\']?(True|False)["\']?.*?confidence["\']?\s*:\s*(\d{1,3})',
            r'confidence["\']?\s*:\s*(\d{1,3}).*?answer["\']?\s*:\s*["\']?(True|False)["\']?'
        ]
        
        for pattern in bool_patterns:
            match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    if groups[0] in ['True', 'False']:
                        result["answer"] = groups[0]
                        result["confidence"] = int(groups[1])
                    else:
                        result["confidence"] = int(groups[0])
                        result["answer"] = groups[1]
                break
        
        # Pattern 3: Extract confidence numbers
        if result["confidence"] is None:
            conf_matches = re.findall(r'confidence["\']?\s*:\s*(\d{1,3})', output, re.IGNORECASE)
            if conf_matches:
                result["confidence"] = int(conf_matches[0])
        
        # Pattern 4: Extract answer values
        if result["answer"] is None:
            answer_matches = re.findall(r'answer["\']?\s*:\s*["\']?([^"\'}\n,]+)["\']?', output, re.IGNORECASE)
            if answer_matches:
                result["answer"] = answer_matches[0].strip()
        
        # Pattern 5: Simple True/False detection
        if result["answer"] is None:
            if re.search(r'\bTrue\b', output, re.IGNORECASE):
                result["answer"] = "True"
            elif re.search(r'\bFalse\b', output, re.IGNORECASE):
                result["answer"] = "False"
        
        return result
    
    def _clean_confidence(self, confidence) -> Union[int, None]:
        """Clean and validate confidence scores."""
        if confidence is None:
            return None
        
        try:
            if isinstance(confidence, str):
                # Extract numbers from string
                numbers = re.findall(r'\d+', confidence)
                if numbers:
                    confidence = int(numbers[0])
                else:
                    return None
            
            confidence = int(confidence)
            
            # Validate range
            if 0 <= confidence <= 100:
                return confidence
            elif confidence > 100:
                # Sometimes models output values > 100, normalize
                return min(confidence, 100)
            else:
                return None
                
        except (ValueError, TypeError):
            return None
    
    def parse_with_llm(self, original_output: str, question: str = "", context: str = "") -> Dict[str, Union[str, int, None]]:
        """
        Use an LLM to parse malformed outputs into proper JSON format.
        """
        if not self.model_wrapper:
            return {"answer": None, "confidence": None}
        
        parsing_prompt = f"""
You are a data parser. Extract the answer and confidence from the following LLM output.

Original Question: {question}
{f"Context: {context}" if context else ""}

LLM Output to Parse:
{original_output}

Extract and return ONLY a JSON object in this exact format:
{{"answer": "extracted_answer", "confidence": extracted_confidence_number}}

Rules:
- For True/False questions, answer should be exactly "True" or "False"
- For other questions, extract the main factual answer
- Confidence should be a number between 0-100
- If you cannot find an answer or confidence, use null

JSON:"""

        try:
            parsed_response = self.model_wrapper.prompt(parsing_prompt, max_new_tokens=100, temperature=0.1)
            return self.clean_and_parse_json(parsed_response)
        except Exception as e:
            print(f"⚠️ LLM parsing failed: {e}")
            return {"answer": None, "confidence": None}


def process_csv_file(
    input_file: str, 
    output_file: str = None,
    model_output_column: str = "model_output",
    use_llm_for_parsing: bool = False,
    model_wrapper = None,
    question_column: str = "question",
    context_column: str = None
):
    """
    Process a CSV file and add parsed answer and confidence columns.
    """
    print(f"📂 Processing: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    if model_output_column not in df.columns:
        print(f"❌ Column '{model_output_column}' not found in CSV")
        return
    
    parser = LLMOutputParser(use_llm_for_parsing, model_wrapper)
    
    parsed_results = []
    failed_count = 0
    
    print("🔍 Parsing outputs...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        output = str(row[model_output_column]) if pd.notna(row[model_output_column]) else ""
        
        # First try rule-based parsing
        result = parser.clean_and_parse_json(output)
        
        # If rule-based parsing fails and LLM parsing is enabled
        if (result["answer"] is None and result["confidence"] is None and 
            use_llm_for_parsing and model_wrapper):
            
            question = str(row[question_column]) if question_column in df.columns else ""
            context = str(row[context_column]) if context_column and context_column in df.columns else ""
            
            result = parser.parse_with_llm(output, question, context)
        
        if result["answer"] is None and result["confidence"] is None:
            failed_count += 1
        
        parsed_results.append(result)
    
    # Add parsed columns
    df["parsed_answer"] = [r["answer"] for r in parsed_results]
    df["parsed_confidence"] = [r["confidence"] for r in parsed_results]
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_parsed{input_path.suffix}")
    
    # Save results
    df.to_csv(output_file, index=False)
    
    # Print statistics
    total_rows = len(df)
    parsed_answers = df["parsed_answer"].notna().sum()
    parsed_confidences = df["parsed_confidence"].notna().sum()
    
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Statistics:")
    print(f"   Total rows: {total_rows}")
    print(f"   Parsed answers: {parsed_answers} ({parsed_answers/total_rows*100:.1f}%)")
    print(f"   Parsed confidences: {parsed_confidences} ({parsed_confidences/total_rows*100:.1f}%)")
    print(f"   Failed to parse: {failed_count} ({failed_count/total_rows*100:.1f}%)")


def batch_process_directory(
    input_dir: str,
    output_dir: str = None,
    file_pattern: str = "*.csv",
    use_llm_for_parsing: bool = False
):
    """
    Process all CSV files in a directory.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory does not exist: {input_dir}")
        return
    
    csv_files = list(input_path.glob(file_pattern))
    if not csv_files:
        print(f"❌ No CSV files found matching pattern: {file_pattern}")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    model_wrapper = None
    if use_llm_for_parsing:
        try:
            from llm_model_wrappers import load_llama  # Import your model loader
            print("🚀 Loading model for LLM-based parsing...")
            model_wrapper = load_llama()
        except Exception as e:
            print(f"⚠️ Could not load model for LLM parsing: {e}")
            use_llm_for_parsing = False
    
    for csv_file in csv_files:
        output_file = None
        if output_dir:
            output_file = str(Path(output_dir) / f"{csv_file.stem}_parsed{csv_file.suffix}")
        
        process_csv_file(
            str(csv_file), 
            output_file,
            use_llm_for_parsing=use_llm_for_parsing,
            model_wrapper=model_wrapper
        )
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Parse LLM outputs to extract answers and confidence scores")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file or directory")
    parser.add_argument("--output", "-o", help="Output CSV file or directory")
    parser.add_argument("--column", "-c", default="model_output", help="Column containing model outputs")
    parser.add_argument("--pattern", "-p", default="*.csv", help="File pattern for batch processing")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for parsing difficult outputs")
    parser.add_argument("--question-col", default="question", help="Question column name")
    parser.add_argument("--context-col", help="Context column name (optional)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        model_wrapper = None
        if args.use_llm:
            try:
                # You'll need to import your model wrapper here
                from llm_model_wrappers import load_llama
                print("🚀 Loading model for LLM-based parsing...")
                model_wrapper = load_llama()
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        
        process_csv_file(
            str(input_path),
            args.output,
            args.column,
            args.use_llm,
            model_wrapper,
            args.question_col,
            args.context_col
        )
    
    elif input_path.is_dir():
        # Process directory
        batch_process_directory(
            str(input_path),
            args.output,
            args.pattern,
            args.use_llm
        )
    
    else:
        print(f"❌ Path does not exist: {args.input}")


if __name__ == "__main__":
    print("🔧 LLM Output Parser - Processing hard-coded files...")
    
    # Hard-coded file paths - modify these to match your specific files
    files_to_process = [
        {
            "input_file": "output/verbalized_confidence_multi_model_full_results/seperate_prompting/trivia_llama_k5_sep.csv",
            "output_file": None,  # Will auto-generate name with "_parsed" suffix
            "model_output_column": "model_output",
            "question_column": "question",
            "context_column": None,
            "use_llm_parsing": True
        },
        {
            "input_file": "output/verbalized_confidence_multi_model_full_results/seperate_prompting/squad_llama_k5_sep.csv",
            "output_file": None,  # Will auto-generate name with "_parsed" suffix
            "model_output_column": "model_output",
            "question_column": "question",
            "context_column": "passage",  # BoolQ has passage context
            "use_llm_parsing": True
        }
        # Add more files here as needed:
        # {
        #     "input_file": "path/to/your/file.csv",
        #     "output_file": "path/to/output/file_parsed.csv",  # or None for auto-generation
        #     "model_output_column": "model_output",
        #     "question_column": "question",
        #     "context_column": "context",  # or None if no context
        #     "use_llm_parsing": True  # Set to True if you want LLM-based parsing for difficult cases
        # }
    ]
    
    # Process each file
    for i, file_config in enumerate(files_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(files_to_process)}")
        print(f"{'='*60}")
        
        if not os.path.exists(file_config["input_file"]):
            print(f"⚠️ File not found: {file_config['input_file']}")
            continue
        
        # Load model wrapper if LLM parsing is requested
        model_wrapper = None
        if file_config.get("use_llm_parsing", False):
            try:
                from llm_model_wrappers import load_llama
                print("🚀 Loading model for LLM-based parsing...")
                model_wrapper = load_llama()
            except Exception as e:
                print(f"⚠️ Could not load model for LLM parsing: {e}")
                print("   Falling back to rule-based parsing only.")
        
        # Process the file
        process_csv_file(
            input_file=file_config["input_file"],
            output_file=file_config["output_file"],
            model_output_column=file_config["model_output_column"],
            use_llm_for_parsing=file_config.get("use_llm_parsing", False),
            model_wrapper=model_wrapper,
            question_column=file_config["question_column"],
            context_column=file_config.get("context_column")
        )
        
        # Clean up model to free memory
        if model_wrapper:
            del model_wrapper
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    print(f"\n✅ Completed processing {len(files_to_process)} files!")
    
    # Uncomment the line below if you want to use command line arguments instead
    # main()