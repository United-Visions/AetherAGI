"""
HumanEval Benchmark Implementation.

OpenAI's code generation benchmark with test case validation.
https://github.com/openai/human-eval
"""
import re
import ast
import subprocess
import tempfile
from typing import List, Tuple, Optional, Any
from pathlib import Path

from ..base import BaseBenchmark, Question, CodeExecutionMixin
from ..config import BenchmarkConfig, get_benchmark


class HumanEvalBenchmark(BaseBenchmark, CodeExecutionMixin):
    """
    HumanEval benchmark implementation.
    
    Tests Python function completion with docstring specification
    and test case validation.
    
    Answer format: Python code (function body)
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None, data_dir: Optional[Path] = None):
        config = config or get_benchmark("humaneval")
        super().__init__(config, data_dir)
        self._dataset = None
    
    def load_questions(self, num_samples: Optional[int] = None) -> List[Question]:
        """Load HumanEval problems from HuggingFace."""
        num_samples = num_samples or self.config.num_samples
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai_humaneval", split="test")
            self._dataset = dataset
            
            questions = []
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # The prompt is the function signature + docstring
                prompt = item["prompt"]
                canonical_solution = item["canonical_solution"]
                test_code = item["test"]
                entry_point = item["entry_point"]
                
                questions.append(Question(
                    id=item["task_id"],
                    text=prompt,
                    expected_answer=canonical_solution,
                    difficulty=self._estimate_difficulty(prompt, canonical_solution),
                    category=self._categorize_problem(prompt),
                    metadata={
                        "test_code": test_code,
                        "entry_point": entry_point,
                        "prompt": prompt,
                        "source": "huggingface",
                    }
                ))
            
            return questions
            
        except ImportError:
            print("HuggingFace datasets not available, using built-in samples")
            return self._load_builtin_samples(num_samples)
    
    def _load_builtin_samples(self, num_samples: int) -> List[Question]:
        """Load built-in sample problems."""
        samples = [
            {
                "prompt": '''def add(a: int, b: int) -> int:
    """Add two integers and return the sum.
    
    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
''',
                "solution": "    return a + b",
                "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0",
                "entry_point": "add",
                "category": "arithmetic",
            },
            {
                "prompt": '''def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome.
    
    >>> is_palindrome("racecar")
    True
    >>> is_palindrome("hello")
    False
    """
''',
                "solution": "    return s == s[::-1]",
                "test": 'assert is_palindrome("racecar") == True\nassert is_palindrome("hello") == False\nassert is_palindrome("") == True',
                "entry_point": "is_palindrome",
                "category": "strings",
            },
            {
                "prompt": '''def factorial(n: int) -> int:
    """Calculate the factorial of n.
    
    >>> factorial(5)
    120
    >>> factorial(0)
    1
    """
''',
                "solution": "    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1",
                "entry_point": "factorial",
                "category": "recursion",
            },
        ]
        
        questions = []
        for i, s in enumerate(samples[:num_samples]):
            questions.append(Question(
                id=f"HumanEval_builtin_{i}",
                text=s["prompt"],
                expected_answer=s["solution"],
                difficulty="medium",
                category=s["category"],
                metadata={
                    "test_code": s["test"],
                    "entry_point": s["entry_point"],
                    "prompt": s["prompt"],
                    "source": "builtin",
                },
            ))
        
        return questions
    
    def check_answer(self, question: Question, response: str) -> Tuple[bool, Any]:
        """
        Check if the code solution passes all test cases.
        
        Uses sandboxed execution to run the code with test cases.
        """
        # Extract code from response
        extracted_code = self.extract_answer(response)
        
        # Build full solution
        prompt = question.metadata.get("prompt", question.text)
        full_code = prompt + extracted_code
        
        # Get test code
        test_code = question.metadata.get("test_code", "")
        
        # Execute and check
        passed, error = self._execute_and_test(full_code, test_code)
        
        return passed, {"code": extracted_code, "error": error if not passed else None}
    
    def extract_answer(self, response: str) -> str:
        """Extract Python code from response."""
        response = self._strip_aether_tags(response)
        
        # Look for code block
        code_match = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Look for indented code (function body)
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                in_code = True
                code_lines.append(line)
            elif in_code and line.strip() == '':
                code_lines.append(line)
            elif in_code and not line.startswith('#'):
                break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Fallback: assume entire response is code
        return response.strip()
    
    def _execute_and_test(self, code: str, test_code: str, timeout: float = 10.0) -> Tuple[bool, str]:
        """Execute code with tests in a sandboxed subprocess."""
        full_code = f"""
{code}

# Run tests
try:
{chr(10).join('    ' + line for line in test_code.split(chr(10)) if line.strip())}
    print("ALL_TESTS_PASSED")
except AssertionError as e:
    print(f"ASSERTION_FAILED: {{e}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                f.flush()
                
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                output = result.stdout + result.stderr
                
                if "ALL_TESTS_PASSED" in output:
                    return True, ""
                elif "ASSERTION_FAILED" in output:
                    return False, output
                else:
                    return False, output
                    
        except subprocess.TimeoutExpired:
            return False, "Timeout: code took too long to execute"
        except Exception as e:
            return False, str(e)
    
    def _estimate_difficulty(self, prompt: str, solution: str) -> str:
        """Estimate problem difficulty."""
        # Count solution complexity
        solution_lines = len([l for l in solution.split('\n') if l.strip()])
        
        # Check for complex patterns
        complex_patterns = ['recursion', 'dynamic', 'tree', 'graph', 'heap', 'binary']
        has_complex = any(p in prompt.lower() for p in complex_patterns)
        
        if has_complex or solution_lines > 15:
            return "hard"
        elif solution_lines > 5:
            return "medium"
        else:
            return "easy"
    
    def _categorize_problem(self, prompt: str) -> str:
        """Categorize the problem type."""
        prompt_lower = prompt.lower()
        
        if any(w in prompt_lower for w in ['list', 'array', 'sort', 'filter']):
            return "lists"
        elif any(w in prompt_lower for w in ['string', 'char', 'text']):
            return "strings"
        elif any(w in prompt_lower for w in ['tree', 'node', 'binary']):
            return "trees"
        elif any(w in prompt_lower for w in ['dict', 'hash', 'map']):
            return "dictionaries"
        elif any(w in prompt_lower for w in ['recurs', 'factorial', 'fibonacci']):
            return "recursion"
        elif any(w in prompt_lower for w in ['math', 'number', 'prime', 'sum']):
            return "math"
        else:
            return "general"
    
    def format_question_for_aether(self, question: Question) -> str:
        """Format question for Aether."""
        return f"""Complete this Python function:

{question.text}

Return only the function body code (indented with 4 spaces).
The function signature and docstring are already provided above.
"""
