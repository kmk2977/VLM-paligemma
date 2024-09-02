#!/bin/bash

MODEL_PATH="paligemma-weights/paligemma-3b-pt-224"
PROMPT="this building is "
IMAGE_FILE_PATH="test_images/b93a1975e9ca4aeea7fc861ef8e88abb.jpeg"
MAX_TOKENS_TO_GENERATE="100"
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python inference.ipynb \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \