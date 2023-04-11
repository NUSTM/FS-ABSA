PLM_DIR='/root/data1/ABSA-Pretraining/outputs/20230210##170350/Global-4500/'

TASK_NAME='ASPE'

for OUTPUT_TYPE in 'span'
do
    for data in '16res'
    do
        for seed in 29 42 757
        do
            for learning_rate in 3e-4
            do
                for batch_size in 8
                do
                    for warmup_rate in 0.06
                    do
                        echo ${data}
                        echo ${seed}
                        echo ${PLM_DIR}
                        echo ${TASK_NAME}
                        CUDA_VISIBLE_DEVICES=0 python ./main.py \
                            --task ${TASK_NAME} \
                            --dataset ${data} \
                            --model_name_or_path ${PLM_DIR} \
                            --seed ${seed} \
                            --output_type ${OUTPUT_TYPE} \
                            --use_french_data \
                            --is_extra_id \
                            --do_train \
                            --do_direct_eval \
                            --train_batch_size ${batch_size} \
                            --gradient_accumulation_steps 1 \
                            --eval_batch_size ${batch_size} \
                            --learning_rate ${learning_rate} \
                            --num_train_epochs 20 \
                            --warmup_rate ${warmup_rate}
                    done
                done
            done
        done
    done
done