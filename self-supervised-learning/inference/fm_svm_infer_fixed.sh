#!/bin/bash

# ========================================
# 데이터 준비: 유효한 클래스 폴더 구조 생성
# ========================================
ORIGINAL_DATA="/workspace/inference/drive-download-20250929T010803Z-1-001"
TEMP_DATA="/workspace/inference/temp_inference_data"

# public_data.py의 self.class_에 정의된 유효한 클래스 이름 중 하나 사용
CLASS_FOLDER="$TEMP_DATA/chestdr"  # 또는 chexpert, mimic 등

echo "Preparing data structure..."

# 임시 디렉토리 생성
rm -rf "$TEMP_DATA"
mkdir -p "$CLASS_FOLDER"

# 심볼릭 링크로 파일 연결
for file in "$ORIGINAL_DATA"/*.dcm; do
    if [ -f "$file" ]; then
        ln -s "$file" "$CLASS_FOLDER/"
    fi
done

echo "✓ Created temporary class structure at: $TEMP_DATA"
echo "✓ Class folder: chestdr (class_id=0)"
echo "✓ Files linked: $(ls -1 "$CLASS_FOLDER" | wc -l)"

# ========================================
# 추론 실행
# ========================================
echo ""
echo "Starting inference..."
echo ""

python fm_svm_infer4.py \
  --config-file /workspace/inference/configs/v2_stable_config.yaml \
  --pretrained-weights /workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth \
  --pretrained-linear-list "/workspace/eval/v2_stable_e400/normal-triage-FM-v4/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v2/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v6/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v12/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v8/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v5/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v10/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v3/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v7/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v13/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v11/running_checkpoint_linear_eval_36250.pth" \
  --test-dataset normal-triage:root="$TEMP_DATA" \
  --batch-size 1 \
  --training-num-classes 3 \
  --svm-model-path /workspace/inference/svm_weight_jh.pickle

EXIT_CODE=$?

# ========================================
# 정리
# ========================================
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference completed successfully!"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
fi

echo "Cleaning up temporary files..."
rm -rf "$TEMP_DATA"
echo "✓ Done!"

exit $EXIT_CODE
