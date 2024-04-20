import argparse
from typing import Dict, List

import yaml

# Different languages that are part of xwinograd.
# These correspond to dataset names (Subsets) on HuggingFace.
# A yaml file is generated by this script for each language.
LANGUAGES = ["en", "fr", "jp", "pt", "ru", "zh"]


def doc_to_text(doc: Dict) -> int:
    """
    Return index of the correct choice.

    Note: We are using the "multiple input" mode of the multiple-choice
        output-type, which means we use different contexts with the same target
        for the different choices, rather than the same context and different targets.
    """
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc: Dict) -> str:
    """
    Return the target completion.

    Note that this does not depend on the correct choice as we are using
    "multiple input" mode.
    """
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc: Dict) -> List[str]:
    """Return the choices that will be used as contexts in "multiple input" mode."""
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


def gen_lang_yamls(output_dir: str, overwrite: bool) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    for lang in LANGUAGES:
        file_name = f"xwinograd_{lang}.yaml"
        try:
            with open(f"{output_dir}/{file_name}", "w" if overwrite else "x") as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": "xwinograd_common_yaml",
                        "dataset_name": lang,
                        "task": f"xwinograd_{lang}",
                    },
                    f,
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default=".", help="Directory to write yaml files to"
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
