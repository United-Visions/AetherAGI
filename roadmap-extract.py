
import os
import re
from loguru import logger

# Configure loguru
logger.add("roadmap_extraction.log", rotation="10 MB")

def extract_and_create_files(source_md_path, target_root_dir, agi_roadmap_dir):
    logger.info(f"Starting extraction from {source_md_path} to {target_root_dir}")
    with open(source_md_path, 'r') as f:
        content = f.read()

    # Create the agi_roadmap directory if it doesn't exist
    try:
        os.makedirs(agi_roadmap_dir, exist_ok=True)
        os.makedirs(os.path.join(agi_roadmap_dir, 'snippets'), exist_ok=True)
        logger.info(f"Ensured directories {agi_roadmap_dir} and {os.path.join(agi_roadmap_dir, 'snippets')} exist.")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return

    # Regex to find each markdown section and its corresponding snippet
    # It captures the markdown file name, its content, the snippet file path, and its content
    pattern = re.compile(
        r'(#\s*\d{2}\s*–\s*.*?\n(?:.|\n)*?)(?P<markdown_filename>\d{2}_.*?\.md)\n(?:Markdown\nCopy\nCode\nPreview\n)?```markdown\n(?P<markdown_content>.*?)\n```(?:\nPython\nCopy\n)?```python\n"""\nPath: (?P<snippet_path>.*?)\n(?P<snippet_content>.*?)"""(?:\nPatch to.*?)?',
        re.DOTALL
    )
    
    # Adjusted pattern to capture all sections correctly
    pattern_all = re.compile(
        r'(?P<md_header>#\s*\d{2}\s*–\s*.*?)\n'  # Markdown header, e.g., # 01 – Persistent Agent Core
        r'((?:.|\n)*?)'  # Any content in between, non-greedy
        r'(?P<md_filename>\d{2}_.*?\.md)\n' # Markdown filename, e.g., 01_state_machine.md
        r'Markdown\nCopy\nCode\nPreview\n```markdown\n'  # Markdown code block start
        r'(?P<md_content>.*?)\n'  # Markdown content, non-greedy
        r'```\n'  # Markdown code block end
        r'Python\nCopy\n'  # Optional Python header
        r'```python\n'  # Python code block start
        r'"""\nPath:\s*(?P<snippet_filepath>.*?)\n'  # Snippet file path
        r'(?P<snippet_content>.*?)'  # Snippet content, non-greedy
        r'"""'
        r'(?:\n(?:Patch to .*?\.py\nPython\nCopy\n)?```python\n.*?```\n)?', # Optional patch section
        re.DOTALL
    )

    matches = pattern_all.finditer(content)

    for match in matches:
        try:
            md_filename = match.group('md_filename').strip()
            md_content = match.group('md_content').strip()
            snippet_filepath = match.group('snippet_filepath').strip()
            snippet_content = match.group('snippet_content').strip()
        except IndexError as e:
            logger.error(f"Regex match failed to find all groups: {e}")
            continue


        # Write markdown file
        try:
            md_target_path = os.path.join(agi_roadmap_dir, md_filename)
            with open(md_target_path, 'w') as f:
                f.write(match.group('md_header').strip() + "\n" + md_content) # Re-add header to content
            logger.info(f"Created: {md_target_path}")
        except IOError as e:
            logger.error(f"Error writing markdown file {md_target_path}: {e}")
            continue

        # Write snippet file
        try:
            snippet_target_path = os.path.join(target_root_dir, snippet_filepath)
            os.makedirs(os.path.dirname(snippet_target_path), exist_ok=True)
            with open(snippet_target_path, 'w') as f:
                f.write('"""\nPath: ' + snippet_filepath + '\n' + snippet_content + '"""')
            logger.info(f"Created: {snippet_target_path}")
        except IOError as e:
            logger.error(f"Error writing snippet file {snippet_target_path}: {e}")
            continue

    # Handle 00_index.md separately as it doesn't follow the same pattern
    index_md_content_match = re.search(
        r'(# AetherMind AGI Upgrade Index.*?)(?=\n01_state_machine\.md)',
        content,
        re.DOTALL
    )
    if index_md_content_match:
        try:
            index_md_content = index_md_content_match.group(1).strip()
            index_md_target_path = os.path.join(agi_roadmap_dir, '00_index.md')
            with open(index_md_target_path, 'w') as f:
                f.write(index_md_content)
            logger.info(f"Created: {index_md_target_path}")
        except IOError as e:
            logger.error(f"Error writing index markdown file {index_md_target_path}: {e}")
    logger.info("Extraction process completed.")

if __name__ == "__main__":
    source_file = "/Users/deion/Desktop/aethermind_universal/add_agi_docs.md"
    target_root = "/Users/deion/Desktop/aethermind_universal"
    agi_roadmap_folder = "/Users/deion/Desktop/aethermind_universal/agi_roadmap"
    extract_and_create_files(source_file, target_root, agi_roadmap_folder)
