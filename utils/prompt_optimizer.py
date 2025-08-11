# -*- coding: utf-8 -*-

"""
Prompt Optimizer for Stable Diffusion

This script provides a lightweight yet effective way to optimize prompts for text-to-image models
like Stable Diffusion. It intelligently extracts texture and structure-related keywords from
a list of raw tags, and then constructs a detailed, high-quality positive prompt.

The core logic uses a predefined dictionary expanded with WordNet's semantic network
to achieve a broad understanding of relevant terms without relying on large language models.

Main interface:
- generate_optimized_prompt(raw_prompt_string: str) -> str
"""

import nltk
from nltk.corpus import wordnet as wn

# --- NLTK WordNet Data Check ---
# Ensures that the necessary WordNet data is available on the system.
try:
    # This line will raise a LookupError if 'wordnet' is not found.
    wn.synsets('dog', pos=wn.NOUN)
except LookupError:
    print("NLTK 'wordnet' package not found. Downloading...")
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual Wordnet, good practice
    print("Download complete.")


class PromptOptimizer:
    """
    Encapsulates the logic for keyword extraction and prompt construction.
    The expensive dictionary expansion is done only once during initialization.
    """

    def __init__(self):
        """
        Initializes the optimizer by building and expanding the keyword dictionaries.
        """
        print("Initializing PromptOptimizer: Building and expanding dictionaries...")
        texture_base = {
            'furry', 'fluffy', 'smooth', 'rough', 'bumpy', 'glossy', 'matte',
            'metallic', 'wooden', 'plastic', 'glassy', 'velvety', 'silky',
            'woolen', 'leathery', 'scaly', 'feathered', 'grainy', 'sandy',
            'rocky', 'watery', 'wet', 'dry', 'cracked', 'polished', 'brushed'
        }

        structure_base = {
            'building', 'house', 'roof', 'window', 'door', 'wall', 'brick', 'tile',
            'fence', 'gate', 'bridge', 'road', 'pillar', 'column', 'arch',
            'scaffolding', 'framework', 'lattice', 'pattern', 'railing', 'stairs',
            'floor', 'ceiling', 'car', 'engine', 'wheel', 'enclosure', 'pen', 'skeleton'
        }

        self.texture_expanded = self._expand_keywords(texture_base, pos=wn.ADJ)
        self.structure_expanded = self._expand_keywords(structure_base, pos=wn.NOUN)
        print("Optimizer is ready.")

    def _expand_keywords(self, keywords: set, pos=None) -> set:
        """
        Expands a set of keywords using WordNet's synonyms and hypernyms.
        """
        expanded = set(keywords)
        for keyword in keywords:
            synsets = wn.synsets(keyword, pos=pos) if pos else wn.synsets(keyword)
            for synset in synsets:
                for lemma in synset.lemmas():
                    expanded.add(lemma.name().lower().replace('_', ' '))
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        expanded.add(lemma.name().lower().replace('_', ' '))
        return expanded

    def extract_keywords(self, raw_tags: list) -> dict:
        """
        Extracts and categorizes keywords from a list of cleaned tags.
        """
        tags = {t.lower().strip() for t in raw_tags if t.strip()}

        texture_tags = sorted([t for t in tags if t in self.texture_expanded])
        structure_tags = sorted([t for t in tags if t in self.structure_expanded])

        all_extracted = set(texture_tags + structure_tags)
        other_tags = sorted([t for t in tags if t not in all_extracted])

        return {"texture": texture_tags, "structure": structure_tags, "others": other_tags}

    def build_positive_prompt(self, extracted_keywords: dict) -> str:
        """
        Constructs the final positive prompt using a three-tiered strategy
        to ensure positive impact and avoid score degradation.
        """
        texture_tags = extracted_keywords["texture"]
        structure_tags = extracted_keywords["structure"]
        other_tags = extracted_keywords["others"]

        total_detail_tags = len(texture_tags) + len(structure_tags)

        base_prompt_str = ", ".join(other_tags)
        quality_boost_str = "masterpiece, best quality, highly detailed, sharp focus, photorealistic"

        # --- 三级优化策略 ---

        # 策略 1: 高可信度 (提取到 >= 3 个细节关键词)
        # 构建详细、具体的增强描述，但使用温和的措辞，避免使用攻击性强的权重。
        if total_detail_tags >= 3:
            detail_parts = []
            # 将 "wooden texture", "furry texture" 改为更自然的 "detailed wood", "fluffy fur"
            for tag in texture_tags:
                detail_parts.append(f"detailed {tag} texture")
            for tag in structure_tags:
                detail_parts.append(f"detailed {tag} structure")

            detail_prompt_str = ", ".join(detail_parts)

            # 最终组合: 基础描述, 细节描述, 通用质量提升
            return f"{base_prompt_str}, {detail_prompt_str}, {quality_boost_str}"

        # 策略 2: 中可信度 (提取到 1-2 个细节关键词)
        # 不指明具体物体，使用通用的、方向性的增强短语。
        elif total_detail_tags > 0:
            generic_enhancement_str = "enhanced textures and fine details"

            # 最终组合: 基础描述, 通用细节增强, 通用质量提升
            return f"{base_prompt_str}, {generic_enhancement_str}, {quality_boost_str}"

        # 策略 3: 最低干预 (未提取到任何细节关键词)
        # 只在原始标签后添加通用质量提升词，确保不产生负面影响。
        else:
            # 最终组合: 基础描述, 通用质量提升
            return f"{base_prompt_str}, {quality_boost_str}"


# --- Singleton Instance ---
# Create a single, reusable instance of the optimizer when the module is loaded.
# This prevents the expensive dictionary-building process from running on every call.
_optimizer_instance = PromptOptimizer()


# --- PUBLIC INTERFACE ---
def generate_optimized_prompt(raw_prompt_string: str) -> str:
    """
    Takes a pipe-separated string of tags and returns an optimized positive prompt.

    This is the main public function to be used by other parts of an application.

    Args:
        raw_prompt_string: A string containing raw tags, separated by '|'.
                           Example: "animal | pig | enclosure | fence | floor"

    Returns:
        A single string containing the optimized positive prompt for Stable Diffusion.
    """
    if not isinstance(raw_prompt_string, str) or not raw_prompt_string.strip():
        # Return a default high-quality prompt for empty or invalid input
        return "masterpiece, best quality, ultra-detailed, sharp focus, 8k, photorealistic"

    # 1. Parse the input string into a list of tags
    raw_tags = [tag.strip() for tag in raw_prompt_string.split('|')]

    # 2. Use the global optimizer instance to extract keywords
    extracted = _optimizer_instance.extract_keywords(raw_tags)

    # 3. Build and return the positive prompt
    positive_prompt = _optimizer_instance.build_positive_prompt(extracted)

    return positive_prompt


# --- Example Usage ---
if __name__ == "__main__":
    # This block will only run when the script is executed directly
    # It demonstrates how to use the `generate_optimized_prompt` function.

    # Example 1: Your provided prompt
    input_prompt_1 = "animal | pig | enclosure | fence | floor | herd | huddle | pen | stand | white"

    print("\n" + "=" * 20 + " EXAMPLE 1 " + "=" * 20)
    print(f"Input String: \n  {input_prompt_1}")

    optimized_prompt_1 = generate_optimized_prompt(input_prompt_1)

    print("\nOptimized Positive Prompt:")
    print(f"  {optimized_prompt_1}")
    print("=" * 53 + "\n")

    # Example 2: A prompt with both texture and structure
    input_prompt_2 = "cat | sitting on a | wooden | bench | fluffy | fur | brick | wall | background"

    print("=" * 20 + " EXAMPLE 2 " + "=" * 20)
    print(f"Input String: \n  {input_prompt_2}")

    optimized_prompt_2 = generate_optimized_prompt(input_prompt_2)

    print("\nOptimized Positive Prompt:")
    print(f"  {optimized_prompt_2}")
    print("=" * 53)