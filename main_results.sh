#!/usr/bin/bash

#SBATCH -J CDFSOD
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v8
#SBATCH -t 6-00:00:00
#SBATCH -o logs/slurm-%A.out

datalist=(
dior
)

shot_list=(
10
)

model_list=(
"l"
#"b"
#"s"
)

for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do
      OUTDIR=output/vit${model}/${dataset}_${shot}shot/
      CFG=configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml

      CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net.py --num-gpus 4 \
        --config-file "${CFG}" \
        MODEL.WEIGHTS /local_datasets/CD_VITO/weights/trained/few-shot/vit${model}_0089999.pth \
        DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR "${OUTDIR}"

      # 학습 산출물 체크포인트 경로 결정 (model_final.pth 없으면 last_checkpoint 사용)
      WEIGHTS="${OUTDIR}/model_final.pth"
      if [ ! -f "${WEIGHTS}" ]; then
        WEIGHTS="$(cat "${OUTDIR}/last_checkpoint")"
      fi

      # 2) EVAL-ONLY (동일 도메인 테스트 1회)
      CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 --eval-only \
        --config-file "${CFG}" \
        MODEL.WEIGHTS "${WEIGHTS}" \
        DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        OUTPUT_DIR "${OUTDIR}/eval"
    done
  done
done


