#!/bin/bash

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

export XFORMERS_DISABLED=1

# 공통 설정
CONFIG_FILE="/workspace/inference/configs/v2_stable_config.yaml"
PRETRAINED_WEIGHTS="/workspace/weights/v2_stable_e400/training_499999/teacher_checkpoint.pth"
LINEAR_LIST="/workspace/eval/v2_stable_e400/normal-triage-FM-v4/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v2/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v6/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v12/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v8/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v5/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v10/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v3/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v7/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v13/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v9/running_checkpoint_linear_eval_36250.pth,/workspace/eval/v2_stable_e400/normal-triage-FM-v11/running_checkpoint_linear_eval_36250.pth"
TEST_DATASET="normal-triage:root=/workspace/dataset/v3/valid"
BATCH_SIZE=1
TRAINING_NUM_CLASSES=3
SVM_MODEL_PATH="/workspace/inference/svm_weight_jh.pickle"

# 결과를 저장할 메인 디렉토리
OUTPUT_BASE="./output"
mkdir -p ${OUTPUT_BASE}

# 전체 결과를 저장할 CSV 파일
SUMMARY_FILE="${OUTPUT_BASE}/threshold_summary.csv"

# CSV 헤더 작성
echo "Threshold,Accuracy,NPV,TP,TN,FP,FN,Precision,Sensitivity,Specificity,PPV,Output_Dir" > ${SUMMARY_FILE}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Threshold Sweep Experiment${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Range: 0.10 to 0.90 (step 0.05)"
echo -e "Total experiments: 17"
echo -e "Summary file: ${SUMMARY_FILE}"
echo ""

# Threshold 배열 (bc 없이 직접 정의)
#thresholds=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90)
thresholds=(0.26 0.27 0.28 0.29)
total=${#thresholds[@]}

# 카운터
count=0

# 각 threshold에 대해 실험
for threshold in "${thresholds[@]}"; do
    count=$((count + 1))
    
    # 출력 디렉토리 이름 생성 (0.10 -> 010, 0.15 -> 015)
    threshold_int=$(echo ${threshold} | sed 's/0\.\([0-9]*\)/\1/')
    threshold_str=$(printf "%03d" ${threshold_int})
    outdir="${OUTPUT_BASE}/svm_out${threshold_str}"
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment ${count}/${total}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Threshold: ${YELLOW}${threshold}${NC}"
    echo -e "Output directory: ${outdir}"
    echo -e "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 디렉토리 생성
    mkdir -p ${outdir}
    
    # 실험 실행
    python fm_svm_infer4_jpg_dcm_with_threshold.py \
        --config-file ${CONFIG_FILE} \
        --pretrained-weights ${PRETRAINED_WEIGHTS} \
        --pretrained-linear-list "${LINEAR_LIST}" \
        --test-dataset ${TEST_DATASET} \
        --batch-size ${BATCH_SIZE} \
        --training-num-classes ${TRAINING_NUM_CLASSES} \
        --svm-model-path ${SVM_MODEL_PATH} \
        --svm-threshold ${threshold} \
        --outdir ${outdir} \
        > ${outdir}/log.txt 2>&1
    
    # 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Experiment completed successfully${NC}"
        
        # 로그에서 metrics 추출
        if [ -f ${outdir}/log.txt ]; then
            accuracy=$(grep "Accuracy:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            npv=$(grep "NPV:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            tp=$(grep "TP:" ${outdir}/log.txt | tail -1 | awk '{print $2}')
            tn=$(grep "TN:" ${outdir}/log.txt | tail -1 | awk '{print $2}')
            fp=$(grep "FP:" ${outdir}/log.txt | tail -1 | awk '{print $2}')
            fn=$(grep "FN:" ${outdir}/log.txt | tail -1 | awk '{print $2}')
            precision=$(grep "Precision:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            sensitivity=$(grep "Sensitivity:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            specificity=$(grep "Specificity:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            ppv=$(grep "PPV:" ${outdir}/log.txt | tail -1 | awk '{print $2}' | sed 's/%//')
            
            # metrics.txt 파일 생성
            cat > ${outdir}/metrics.txt << EOF
Threshold: ${threshold}
Accuracy: ${accuracy}%
NPV: ${npv}%
TP: ${tp}
TN: ${tn}
FP: ${fp}
FN: ${fn}
Precision: ${precision}%
Sensitivity: ${sensitivity}%
Specificity: ${specificity}%
PPV: ${ppv}%
EOF
            
            echo -e "Metrics saved to: ${outdir}/metrics.txt"
            
            # Summary CSV에 추가
            echo "${threshold},${accuracy},${npv},${tp},${tn},${fp},${fn},${precision},${sensitivity},${specificity},${ppv},${outdir}" >> ${SUMMARY_FILE}
            
            # 결과 미리보기
            echo -e "\n${YELLOW}Results Preview:${NC}"
            echo -e "  Accuracy: ${accuracy}%"
            echo -e "  Sensitivity: ${sensitivity}%"
            echo -e "  Specificity: ${specificity}%"
            echo -e "  PPV: ${ppv}%"
            echo -e "  NPV: ${npv}%"
        fi
    else
        echo -e "${RED}✗ Experiment failed${NC}"
        echo -e "Check log file: ${outdir}/log.txt"
        
        # 에러 메시지 출력
        if [ -f ${outdir}/log.txt ]; then
            echo -e "\n${RED}Error log (last 10 lines):${NC}"
            tail -10 ${outdir}/log.txt
        fi
        
        # Summary에 실패 기록
        echo "${threshold},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,${outdir}" >> ${SUMMARY_FILE}
    fi
    
    echo -e "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 마지막 실험이 아니면 대기
    if [ ${count} -lt ${total} ]; then
        echo -e "${YELLOW}Waiting 5 seconds before next experiment...${NC}"
        sleep 5
        echo ""
    fi
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Experiments Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Total experiments: ${total}"
echo -e "Summary file: ${SUMMARY_FILE}"
echo ""

# Summary 테이블 출력 (column 명령어가 없을 수 있으므로 python 사용)
echo -e "${YELLOW}Summary Table:${NC}"
if command -v column &> /dev/null; then
    column -t -s',' ${SUMMARY_FILE}
else
    cat ${SUMMARY_FILE}
fi
echo ""

# 최고 성능 찾기 (python 사용)
python << 'PYTHON_SCRIPT'
import csv
import sys

try:
    with open('./output/threshold_summary.csv', 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row['Accuracy'] != 'FAILED']
    
    if not rows:
        print("No successful experiments found.")
        sys.exit(0)
    
    # 숫자로 변환
    for row in rows:
        for key in ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'Precision']:
            if key in row:
                try:
                    row[key] = float(row[key])
                except:
                    row[key] = 0.0
    
    print("\n\033[1;33mFinding best thresholds...\033[0m\n")
    
    # Best Accuracy
    best_acc = max(rows, key=lambda x: x['Accuracy'])
    print(f"\033[0;32mBest Accuracy:\033[0m")
    print(f"  Threshold: {best_acc['Threshold']}, Accuracy: {best_acc['Accuracy']:.2f}%")
    
    # Best Sensitivity
    best_sens = max(rows, key=lambda x: x['Sensitivity'])
    print(f"\033[0;32mBest Sensitivity:\033[0m")
    print(f"  Threshold: {best_sens['Threshold']}, Sensitivity: {best_sens['Sensitivity']:.2f}%")
    
    # Best Specificity
    best_spec = max(rows, key=lambda x: x['Specificity'])
    print(f"\033[0;32mBest Specificity:\033[0m")
    print(f"  Threshold: {best_spec['Threshold']}, Specificity: {best_spec['Specificity']:.2f}%")
    
    # Best PPV
    best_ppv = max(rows, key=lambda x: x['PPV'])
    print(f"\033[0;32mBest PPV:\033[0m")
    print(f"  Threshold: {best_ppv['Threshold']}, PPV: {best_ppv['PPV']:.2f}%")
    
    # Best NPV
    best_npv = max(rows, key=lambda x: x['NPV'])
    print(f"\033[0;32mBest NPV:\033[0m")
    print(f"  Threshold: {best_npv['Threshold']}, NPV: {best_npv['NPV']:.2f}%")
    
except Exception as e:
    print(f"Error analyzing results: {e}")
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}Done! Check ${OUTPUT_BASE}/ for all results.${NC}"
