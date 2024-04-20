import subprocess


def eval_ppl(
    model_name,
    task_list=[
        "wikitext",
    ],
    output_dir=".",
):
    command = "lm_eval"
    tasks = ",".join(task_list)
    args = [
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_name},cache_dir=./.cache,device_map=auto",
        "--tasks",
        f"{tasks}",
        "--device",
        "cuda:0",
        "--batch_size",
        "8",
        "--output_path",
        f"{output_dir}/ppl.json",
    ]
    # Combine command and arguments
    full_command = [command] + args

    # Execute the command
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
