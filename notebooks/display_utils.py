"""
Utilities for displaying code and information in Jupyter notebooks.
"""

from IPython.display import Markdown, display
import inspect
import pathlib
import ast


def display_task_source(task_func, title=None):
    """Display task source code from the original file, including decorators but excluding docstrings.
    
    Only shows code from @flowtask decorator to end of function, excluding the function's docstring.
    
    Args:
        task_func: The function to display source code for
        title: Optional title (kept for backwards compatibility, not currently used)
    """
    # Get the module where the function is defined
    module = inspect.getmodule(task_func)
    func_name = task_func.__name__
    
    if not module or not hasattr(module, '__file__'):
        display(Markdown(f"❌ Could not find source file for `{func_name}`"))
        return
    
    # Read the entire source file
    source_file = pathlib.Path(module.__file__)
    with open(source_file, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    # Parse the AST to find the function
    tree = ast.parse(file_content)
    
    # Find the function definition in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Get line numbers (1-indexed)
            start_line = node.lineno
            end_line = node.end_lineno
            
            # Read the lines
            lines = file_content.splitlines()
            
            # Look backwards from function def to find @flowtask decorator
            decorator_start = start_line - 1  # Convert to 0-indexed
            while decorator_start > 0:
                line = lines[decorator_start - 1].strip()
                if line.startswith('@flowtask'):
                    break
                decorator_start -= 1
            
            # Extract from decorator to end of function
            source_lines = lines[decorator_start - 1:end_line]
            
            # Now remove the docstring if present
            # The docstring is the first statement in the function body
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[0].value.value, str):
                    # Found a docstring - get its line range
                    docstring_start = node.body[0].lineno - 1  # Convert to 0-indexed
                    docstring_end = node.body[0].end_lineno - 1  # Convert to 0-indexed
                    
                    # Calculate relative positions in source_lines array
                    relative_doc_start = docstring_start - (decorator_start - 1)
                    relative_doc_end = docstring_end - (decorator_start - 1)
                    
                    # Remove docstring lines from source_lines
                    source_lines = source_lines[:relative_doc_start] + source_lines[relative_doc_end + 1:]
            
            source_code = '\n'.join(source_lines)
            
            # Display as markdown
            markdown_content = f"""---

### `{func_name}`

```python
{source_code}
```
"""
            display(Markdown(markdown_content))
            return
    
    # Fallback if AST parsing fails
    display(Markdown(f"❌ Could not parse function `{func_name}` from source file"))
